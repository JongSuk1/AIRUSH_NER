import os
from nsml import DATASET_PATH, IS_ON_NSML, SESSION_NAME

os.environ["HF_HOME"] = "/home/nsml/.cache/huggingface"
import nsml
from larva import LarvaTokenizer, LarvaModel
from torch.optim import AdamW
from tqdm import tqdm
import torch
import os
import sys
import argparse
from dataset import MenuOptionDataset
from test_dataset import MenuOptionTestDataset
from model import BertCrfForAttrLSTM, BertCrfForAttrGRU
import torch_optimizer as optim
from torch.cuda import amp
from madgrad import MADGRAD
from torchcrf import CRF
import torch.nn as nn
import numpy as np


def get_dataloader(tokenizer):
    print(DATASET_PATH)
    import glob
    print(glob.glob(f'{DATASET_PATH}/*/*/*'))
    train_dataset = MenuOptionDataset(tokenizer, f'{DATASET_PATH}/train/train_data')
    label_to_idx = train_dataset.get_label_to_idx()

    data_loader = torch.utils.data.DataLoader(
        train_dataset, args.batch_size, num_workers=args.workers)

    return data_loader, label_to_idx


def convert_preds(preds, tokenizer, idx_to_label, attributes):
    convert_preds = []
    for idx, pred in enumerate(preds):
        attribute = attributes[idx]
        attribute = [s for s in attribute[1:] if s != 0][:-1]
        tok = tokenizer.convert_ids_to_tokens(attribute)
        attribute_name = tokenizer.convert_tokens_to_string(tok)
        attribute_name = attribute_name.replace(' ', '')
        pred = [f"{idx_to_label[s]}-{attribute_name}" if idx_to_label[s]
                                                         in ['B', 'I', 'S'] else idx_to_label[s] for s in pred]
        convert_preds.append(pred)

    return convert_preds


def bind_nsml(model, args):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])

    def infer(root_path):
        tokenizer = LarvaTokenizer.from_pretrained('larva-jpn-plus-base-cased', do_lower_case=True)
        test_dataset = MenuOptionTestDataset(tokenizer, root_path)
        label_to_idx = test_dataset.get_label_to_idx()
        device = torch.device('cuda')
        data_loader_val = torch.utils.data.DataLoader(
            test_dataset, args.batch_size, num_workers=args.workers, shuffle=False)
        idx_to_label = dict((y, x) for x, y in label_to_idx.items())
        return _infer(model, data_loader_val, tokenizer, idx_to_label, device)

    nsml.bind(save=save, load=load, infer=infer, use_nsml_legacy=False)


@torch.no_grad()
def _infer(model, data_loader, tokenizer, idx_to_label, device):
    model.eval()
    tqdm_dataloader = tqdm(data_loader)

    tot_preds = []
    for targets in tqdm_dataloader:
        m_input_ids = targets['m_input_ids'].to(device)
        m_input_mask = targets['m_input_mask'].to(device)
        e_input_ids = targets['e_input_ids'].to(device)
        e_input_mask = targets['e_input_mask'].to(device)
        inputs = {"m_input_ids": m_input_ids,
                  'e_input_ids': e_input_ids,
                  "m_attention_mask": m_input_mask,
                  'e_attention_mask': e_input_mask
                  }
        preds = model(**inputs)
        attributes = e_input_ids.to('cpu').tolist()
        preds = convert_preds(preds, tokenizer, idx_to_label, attributes)
        tot_preds += preds

    return tot_preds


class EnsembleModule(nn.Module):
    def __init__(self, model1, model2, model3, model4, model5, model6):
        super(EnsembleModule, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6

    def forward(self, **inputs):
        sot1, logit1 = self.model1(**inputs)
        sot2, logit2 = self.model2(**inputs)
        sot3, logit3 = self.model3(**inputs)
        #sot4, logit4 = self.model4(**inputs)
        #sot5, logit5 = self.model5(**inputs)
        #sot6, logit6 = self.model6(**inputs)

        sot1 = np.array(sot1)
        sot2 = np.array(sot2)
        sot3 = np.array(sot3)
        #sot4 = np.array(sot4)
        #sot5 = np.array(sot5)
        #sot6 = np.array(sot6)

        sot = np.stack([sot1,sot2,sot3])
        #sot = np.stack([sot1,sot2,sot3,sot4,sot5,sot6])

        total_pred = []
        for i in range(len(sot[0])):
            pred = []
            for j in range(len(sot[0][0])):
                vote_list = sot[:,i,j]
                pred.append(np.argmax(np.bincount(vote_list)))
            total_pred.append(pred)
        return total_pred

    def freeze(self):
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False
        for param in self.model3.parameters():
            param.requires_grad = False


def load_nsml_model(model, args, check=None, sess=None):
    bind_nsml(model, args)
    if args.pause:
        nsml.paused(scope=locals())
    nsml.load(checkpoint=check, session=sess)
    return model.to(args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--optimizer", type=str, default='0')
    parser.add_argument("--model", type=str, default='0')

    args = parser.parse_args()
    backbone1 = LarvaModel.from_pretrained('larva-jpn-plus-base-cased')
    model1 = BertCrfForAttrLSTM(backbone1, 7).to(args.device)

    backbone2 = LarvaModel.from_pretrained('larva-jpn-plus-base-cased')
    model2 = BertCrfForAttrLSTM(backbone2, 7).to(args.device)

    backbone3 = LarvaModel.from_pretrained('larva-jpn-plus-base-cased')
    model3 = BertCrfForAttrLSTM(backbone3, 7).to(args.device)

    backbone4 = LarvaModel.from_pretrained('larva-jpn-plus-base-cased')
    model4 = BertCrfForAttrLSTM(backbone4, 7).to(args.device)

    backbone5 = LarvaModel.from_pretrained('larva-jpn-plus-base-cased')
    model5 = BertCrfForAttrLSTM(backbone5, 7).to(args.device)

    backbone6 = LarvaModel.from_pretrained('larva-jpn-plus-base-cased')
    model6 = BertCrfForAttrLSTM(backbone6, 7).to(args.device)

    if args.mode == 'train':
        model1 = load_nsml_model(model1, args, check="74", sess="KR95390/airush2021-2-6a/84")
        model2 = load_nsml_model(model2, args, check="79", sess="KR95390/airush2021-2-6a/84")
        model3 = load_nsml_model(model3, args, check="85", sess="KR95390/airush2021-2-6a/84")
        model4 = load_nsml_model(model4, args, check="53", sess="KR95390/airush2021-2-6a/84")
        model5 = load_nsml_model(model5, args, check="69", sess="KR95390/airush2021-2-6a/84")
        model6 = load_nsml_model(model6, args, check="72", sess="KR95390/airush2021-2-6a/84")

        model = EnsembleModule(model1, model2, model3, model4, model5, model6).to(args.device)
        '''
        tokenizer = LarvaTokenizer.from_pretrained('larva-jpn-plus-base-cased', do_lower_case=True)
        data_loader, label_to_idx = get_dataloader(tokenizer)
        idx_to_label = dict((y, x) for x, y in label_to_idx.items())
        
        for idx, targets in enumerate(data_loader):
            m_input_ids = targets['m_input_ids'].to(args.device)
            m_input_mask = targets['m_input_mask'].to(args.device)
            e_input_ids = targets['e_input_ids'].to(args.device)
            e_input_mask = targets['e_input_mask'].to(args.device)
            inputs = {"m_input_ids": m_input_ids,
                      'e_input_ids': e_input_ids,
                      "m_attention_mask": m_input_mask,
                      'e_attention_mask': e_input_mask,
                      }
            with torch.no_grad():
                pred = model(**inputs)
            print(len(pred))
            print(len(pred[0]))
            
            print(len(sot1))
            print(len(sot1[0]))
            print(len(sot2))
            print(len(sot2[0]))
            print(len(sot3))
            print(len(sot3[0]))
            print(sot1[0])
            print(sot2[0])
            print(sot3[0])
            
            break
            '''
        bind_nsml(model, args)
        if args.pause:
            nsml.paused(scope=locals())
        nsml.save('0')

    else:
        model = EnsembleModule(model1, model2, model3, model4, model5, model6).to(args.device)
        bind_nsml(model, args)
        if args.pause:
            nsml.paused(scope=locals())