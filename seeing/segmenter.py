# Usage as a simple differentiable segmenter base class

import os, torch, numpy, json, glob
import skimage.morphology
from collections import OrderedDict
from . import upsegmodel
from urllib.request import urlretrieve

class BaseSegmenter:
    def get_label_and_category_names(self):
        '''
        Returns two lists: first, a list of tuples [(label, category), ...]
        where the label and category are human-readable strings indicating
        the meaning of a segmentation class.  The 0th segmentation class
        should be reserved for a label ('-') that means "no prediction."
        The second list should just be a list of [category,...] listing
        all categories in a canonical order.
        '''
        raise NotImplemented()

    def segment_batch(self, tensor_images, downsample=1):
        '''
        Returns a multilabel segmentation for the given batch of (RGB [-1...1])
        images.  Each pixel of the result is a torch.long indicating a
        predicted class number.  Multiple classes can be predicted for
        the same pixel: output shape is (n, multipred, y, x), where
        multipred is 3, 5, or 6, for how many different predicted labels can
        be given for each pixel (depending on whether subdivision is being
        used).  If downsample is specified, then the output y and x dimensions
        are downsampled from the original image.
        '''
        raise NotImplemented()

class UnifiedParsingSegmenter(BaseSegmenter):
    '''
    This is a wrapper for a more complicated multi-class segmenter,
    as described in https://arxiv.org/pdf/1807.10221.pdf, and as
    released in https://github.com/CSAILVision/unifiedparsing.
    For our purposes and to simplify processing, we do not use
    whole-scene predictions, and we only consume part segmentations
    for the three largest object classes (sky, building, person).
    '''

    def __init__(self, segsizes=None):
        # Create a segmentation model
        if segsizes is None:
            segsizes = [256]
        segvocab = 'upp'
        segarch = ('resnet50', 'upernet')
        epoch = 40
        ensure_upp_segmenter_downloaded('datasets/segmodel')
        segmodel = load_unified_parsing_segmentation_model(
                segarch, segvocab, epoch)
        segmodel.cuda()
        self.segmodel = segmodel
        self.segsizes = segsizes
        # Assign class numbers for parts.
        first_partnumber = (1 +
                (len(segmodel.labeldata['object']) - 1) +
                (len(segmodel.labeldata['material']) - 1))
        partobjects = segmodel.labeldata['object_part'].keys()
        partnumbers = {}
        partnames = []
        objectnumbers = {k: v
                for v, k in enumerate(segmodel.labeldata['object'])}
        part_index_translation = []
        # We merge some classes.  For example "door" is both an object
        # and a part of a building.  To avoid confusion, we just count
        # such classes as objects, and add part scores to the same index.
        for owner in partobjects:
            part_list = segmodel.labeldata['object_part'][owner]
            numeric_part_list = []
            for part in part_list:
                if part in objectnumbers:
                    numeric_part_list.append(objectnumbers[part])
                elif part in partnumbers:
                    numeric_part_list.append(partnumbers[part])
                else:
                    partnumbers[part] = len(partnames) + first_partnumber
                    partnames.append(part)
                    numeric_part_list.append(partnumbers[part])
            part_index_translation.append(torch.tensor(numeric_part_list))
        self.objects_with_parts = [objectnumbers[obj] for obj in partobjects]
        self.part_index = part_index_translation
        self.part_names = partnames
        # For now we'll just do object and material labels.
        self.num_classes = 1 + (
                len(segmodel.labeldata['object']) - 1) + (
                len(segmodel.labeldata['material']) - 1) + len(partnames)
        self.num_object_classes = len(self.segmodel.labeldata['object']) - 1

    def get_label_and_category_names(self, dataset=None):
        '''
        Lists label and category names.
        '''
        # Labels are ordered as follows:
        # 0, [object labels] [divided object labels] [materials] [parts]
        # The zero label is reserved to mean 'no prediction'.
        suffixes = []
        divided_labels = []
        for suffix in suffixes:
            divided_labels.extend([('%s-%s' % (label, suffix), 'part')
                for label in self.segmodel.labeldata['object'][1:]])
        # Create the whole list of labels
        labelcats = (
                [(label, 'object')
                    for label in self.segmodel.labeldata['object']] +
                divided_labels +
                [(label, 'material')
                    for label in self.segmodel.labeldata['material'][1:]] +
                [(label, 'part') for label in self.part_names])
        return labelcats, ['object', 'part', 'material']

    def raw_seg_prediction(self, tensor_images, downsample=1):
        '''
        Generates a segmentation by applying multiresolution voting on
        the segmentation model, using (rounded to 32 pixels) a set of
        resolutions in the example benchmark code.
        '''
        y, x = tensor_images.shape[2:]
        b = len(tensor_images)
        tensor_images = (tensor_images + 1) / 2 * 255
        tensor_images = torch.flip(tensor_images, (1,)) # BGR!!!?
        tensor_images -= torch.tensor([102.9801, 115.9465, 122.7717]).to(
                   dtype=tensor_images.dtype, device=tensor_images.device
                   )[None,:,None,None]
        seg_shape = (y // downsample, x // downsample)
        # We want these to be multiples of 32 for the model.
        sizes = [(s, s) for s in self.segsizes]
        pred = {category: torch.zeros(
            len(tensor_images), len(self.segmodel.labeldata[category]),
            seg_shape[0], seg_shape[1]).cuda()
            for category in ['object', 'material']}
        part_pred = {partobj_index: torch.zeros(
            len(tensor_images), len(partindex),
            seg_shape[0], seg_shape[1]).cuda()
            for partobj_index, partindex in enumerate(self.part_index)}
        for size in sizes:
            if size == tensor_images.shape[2:]:
                resized = tensor_images
            else:
                resized = torch.nn.AdaptiveAvgPool2d(size)(tensor_images)
            r_pred = self.segmodel(
                dict(img=resized), seg_size=seg_shape)
            for k in pred:
                pred[k] += r_pred[k]
            for k in part_pred:
                part_pred[k] += r_pred['part'][k]
        return pred, part_pred

    def segment_batch(self, tensor_images, downsample=1):
        '''
        Returns a multilabel segmentation for the given batch of (RGB [-1...1])
        images.  Each pixel of the result is a torch.long indicating a
        predicted class number.  Multiple classes can be predicted for
        the same pixel: output shape is (n, multipred, y, x), where
        multipred is 3, 5, or 6, for how many different predicted labels can
        be given for each pixel (depending on whether subdivision is being
        used).  If downsample is specified, then the output y and x dimensions
        are downsampled from the original image.
        '''
        pred, part_pred = self.raw_seg_prediction(tensor_images,
                downsample=downsample)
        y, x = tensor_images.shape[2:]
        seg_shape = (y // downsample, x // downsample)
        segs = torch.zeros(len(tensor_images), 3, # objects, materials, parts
                seg_shape[0], seg_shape[1],
                dtype=torch.long, device=tensor_images.device)
        _, segs[:,0] = torch.max(pred['object'], dim=1)
        # Get materials and translate to shared numbering scheme
        _, segs[:,1] = torch.max(pred['material'], dim=1)
        maskout = (segs[:,1] == 0)
        segs[:,1] += (len(self.segmodel.labeldata['object']) - 1)
        segs[:,1][maskout] = 0
        # Now deal with subparts of sky, buildings, people
        for i, object_index in enumerate(self.objects_with_parts):
            trans = self.part_index[i].to(segs.device)
            # Get the argmax, and then translate to shared numbering scheme
            seg = trans[torch.max(part_pred[i], dim=1)[1]]
            # Only trust the parts where the prediction also predicts the
            # owning object.
            mask = (segs[:,0] == object_index)
            segs[:,2][mask] = seg[mask]
        return segs

def load_unified_parsing_segmentation_model(segmodel_arch, segvocab, epoch):
    segmodel_dir = 'datasets/segmodel/%s-%s-%s' % ((segvocab,) + segmodel_arch)
    # Load json of class names and part/object structure
    with open(os.path.join(segmodel_dir, 'labels.json')) as f:
        labeldata = json.load(f)
    nr_classes={k: len(labeldata[k])
                for k in ['object', 'scene', 'material']}
    nr_classes['part'] = sum(len(p) for p in labeldata['object_part'].values())
    # Create a segmentation model
    segbuilder = upsegmodel.ModelBuilder()
    # example segmodel_arch = ('resnet101', 'upernet')
    seg_encoder = segbuilder.build_encoder(
            arch=segmodel_arch[0],
            fc_dim=2048,
            weights=os.path.join(segmodel_dir, 'encoder_epoch_%d.pth' % epoch))
    seg_decoder = segbuilder.build_decoder(
            arch=segmodel_arch[1],
            fc_dim=2048, use_softmax=True,
            nr_classes=nr_classes,
            weights=os.path.join(segmodel_dir, 'decoder_epoch_%d.pth' % epoch))
    segmodel = upsegmodel.SegmentationModule(
            seg_encoder, seg_decoder, labeldata)
    segmodel.categories = ['object', 'part', 'material']
    segmodel.eval()
    return segmodel

def ensure_upp_segmenter_downloaded(directory):
    baseurl = 'http://netdissect.csail.mit.edu/data/segmodel'
    dirname = 'upp-resnet50-upernet'
    files = ['decoder_epoch_40.pth', 'encoder_epoch_40.pth', 'labels.json']
    download_dir = os.path.join(directory, dirname)
    os.makedirs(download_dir, exist_ok=True)
    for fn in files:
        if os.path.isfile(os.path.join(download_dir, fn)):
            continue # Skip files already downloaded
        url = '%s/%s/%s' % (baseurl, dirname, fn)
        print('Downloading %s' % url)
        urlretrieve(url, os.path.join(download_dir, fn))
    assert os.path.isfile(os.path.join(directory, dirname, 'labels.json'))

def test_main():
    '''
    Test the unified segmenter.
    '''
    from PIL import Image
    testim = Image.open('script/testdata/test_church_242.jpg')
    tensor_im = (torch.from_numpy(numpy.asarray(testim)).permute(2, 0, 1)
            .float() / 255 * 2 - 1)[None, :, :, :].cuda()
    segmenter = UnifiedParsingSegmenter()
    seg = segmenter.segment_batch(tensor_im)
    bc = torch.bincount(seg.view(-1))
    labels, cats = segmenter.get_label_and_category_names()
    for label in bc.nonzero()[:,0]:
        if label.item():
            # What is the prediction for this class?
            pred, mask = segmenter.predict_single_class(tensor_im, label.item())
            assert mask.sum().item() == bc[label].item()
            assert len(((seg == label).max(1)[0] - mask).nonzero()) == 0
            inside_pred = pred[mask].mean().item()
            outside_pred = pred[~mask].mean().item()
            print('%s (%s, #%d): %d pixels, pred %.2g inside %.2g outside' %
                (labels[label.item()] + (label.item(), bc[label].item(),
                    inside_pred, outside_pred)))

if __name__ == '__main__':
    test_main()
