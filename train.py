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
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch_ema import ExponentialMovingAverage



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
        idx_to_label = dict((y,x) for x,y in label_to_idx.items())
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
        preds, _ = model(**inputs)
        attributes = e_input_ids.to('cpu').tolist()
        preds = convert_preds(preds, tokenizer, idx_to_label, attributes)
        tot_preds += preds
        
        
    return tot_preds


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, ema, scheduler=None):
    model.train()
    model.zero_grad()
    print_len = len(data_loader) // 10
    total_batch = 0
    total_loss = 0
    iter_loss = 0
    for idx, targets in enumerate(data_loader):

        m_input_ids = targets['m_input_ids'].to(device)
        m_input_mask = targets['m_input_mask'].to(device)
        e_input_ids = targets['e_input_ids'].to(device)
        e_input_mask = targets['e_input_mask'].to(device)
        labels = targets['label_ids'].to(device)
        inputs = {"m_input_ids": m_input_ids,
                  'e_input_ids': e_input_ids,
                  "m_attention_mask": m_input_mask,
                  'e_attention_mask': e_input_mask,
                  "labels": labels,
                  }
        total_batch += len(m_input_ids)

        with amp.autocast(enabled = args.amp):
            log_likelihood, soq, _  = model(**inputs)

        loss = -1 * log_likelihood
        loss = loss.mean()
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model.parameters())

        #scheduler.step()

        total_loss += loss.item()
        iter_loss += loss.item()
        if idx % print_len == 0:
            #print(f'epoch {epoch + 1}, iter {idx}/{len(data_loader)}: lr {scheduler.get_lr()[0]}')
            print(f"Epoch {epoch + 1}, iter is {idx}/{len(data_loader)} loss {iter_loss}")
            iter_loss = 0
    print(f"Epoch {epoch+1} total loss = {total_loss}")


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
    parser.add_argument('--scheduler', type=str, default="0")
    parser.add_argument("--ema", action="store_true")

    args = parser.parse_args()
    if ~args.ema:
        ema = None

    backbone = LarvaModel.from_pretrained('larva-jpn-plus-base-cased')
    if args.model == 'LSTM':
        model = BertCrfForAttrLSTM(backbone, 7)
    elif args.model == 'GRU':
        model = BertCrfForAttrGRU(backbone, 7)
    model.to(args.device)
    
    bind_nsml(model, args)
    if args.pause:
        nsml.paused(scope=locals())
        
    nsml.load(checkpoint="7", session="KR95390/airush2021-2-6a/164")
    #print('continue training 84-85 for half learning rate')
    if args.mode == 'train':
        tokenizer = LarvaTokenizer.from_pretrained('larva-jpn-plus-base-cased', do_lower_case=True)
        data_loader, label_to_idx = get_dataloader(tokenizer)
        idx_to_label = dict((y,x) for x,y in label_to_idx.items())
        

        #optimizer = AdamW(model.parameters(), lr=args.lr)
        if args.optimizer =='RAdam':
            optimizer = optim.RAdam(model.parameters(),
                                lr = args.lr, betas = (0.9,0.999), eps = 1e-8, weight_decay=0)
        if args.optimizer =='AdamW':
            optimizer = AdamW(model.parameters(), lr=args.lr)
        if args.optimizer == 'MADGRAD':
            optimizer = MADGRAD(model.parameters(),lr=args.lr, momentum=0.99, weight_decay=0, eps=1e-6)
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=args.lr, momentum=0, nesterov=False, weight_decay=1e-4)

        scaler = amp.GradScaler(enabled=args.amp)

        #scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader), epochs=args.epochs,
        #                       pct_start=0.1)
        if args.ema:
            ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

        for epoch in range(args.epochs):
            train_one_epoch(model, optimizer, data_loader, args.device, epoch, scaler, ema=ema, scheduler=None)
            if epoch % args.save_epoch == 0:
                if args.ema:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())

                nsml.save(str(epoch + 1))

                if args.ema:
                    ema.restore(model.parameters())

            #if epoch ==0:
            #    nsml.save(str(epoch+1))
            #if (epoch+1) % 3 == 0:
            #    nsml.save(str(epoch + 1))
