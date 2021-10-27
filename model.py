import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torchcrf import CRF

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()

    def forward(self, title_output, attr_output):
        '''
        title_output (batchsize, seqlen, hidden_dim)
        attr_output (batchsize, hidden_dim)
        '''
        seq_len = title_output.size()[1]
        attr_output = attr_output.unsqueeze(1).repeat(1,seq_len,1)
        cos_sim = torch.cosine_similarity(attr_output,title_output,-1)
        cos_sim = cos_sim.unsqueeze(-1)
        outputs = title_output*cos_sim
        return outputs

class BertCrfForAttrLSTM(nn.Module):
    def __init__(self, bert, num_tags):
        super(BertCrfForAttrLSTM, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        hidden_size = 768
        self.t_lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size // 2,
                                batch_first=True,bidirectional=True)
        self.a_lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size // 2,
                                batch_first=True,bidirectional=True)
        self.attention = Attention()
        self.ln = LayerNorm(hidden_size* 2)
        self.classifier = nn.Linear(hidden_size* 2, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)

    def forward(self, m_input_ids,e_input_ids, m_attention_mask=None,
                 e_attention_mask=None,labels=None):
        # bert
        outputs_title = self.bert(input_ids=m_input_ids, attention_mask=m_attention_mask)
        outputs_attr = self.bert(input_ids=e_input_ids, attention_mask=e_attention_mask)
        # bilstm
        title_output, _ = self.t_lstm(outputs_title[0])
        _, attr_hidden = self.a_lstm(outputs_attr[0])
        # attention
        attr_output = torch.cat([attr_hidden[0][-2],attr_hidden[0][-1]], dim=-1)
        attention_output = self.attention(title_output,attr_output)
        # concat
        outputs = torch.cat([title_output, attention_output], dim=-1)
        outputs = self.ln(outputs)
        sequence_output = self.dropout(outputs)
        logits = self.classifier(sequence_output)
        #print(m_input_ids.shape) 100,80
        #print(logits.shape) 100,80,7
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(logits, labels), self.crf.decode(logits)
            return log_likelihood, sequence_of_tags, logits
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags, logits

class BertCrfForAttrGRU(nn.Module):
    def __init__(self, bert, num_tags):
        super(BertCrfForAttrGRU, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        hidden_size = 768
        self.t_lstm = nn.GRU(input_size=hidden_size,hidden_size=hidden_size // 2,
                                batch_first=True,bidirectional=True)
        self.a_lstm = nn.GRU(input_size=hidden_size,hidden_size=hidden_size // 2,
                                batch_first=True,bidirectional=True)
        self.attention = Attention()
        self.ln = LayerNorm(hidden_size* 2)
        self.classifier = nn.Linear(hidden_size* 2, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)

    def forward(self, m_input_ids,e_input_ids, m_attention_mask=None,
                 e_attention_mask=None,labels=None):
        # bert
        outputs_title = self.bert(input_ids=m_input_ids, attention_mask=m_attention_mask)
        outputs_attr = self.bert(input_ids=e_input_ids, attention_mask=e_attention_mask)
        # bilstm
        title_output, _ = self.t_lstm(outputs_title[0])
        _, attr_hidden = self.a_lstm(outputs_attr[0])
        # attention
        attr_output = torch.cat([attr_hidden[-2], attr_hidden[-1]], dim=-1)
        attention_output = self.attention(title_output,attr_output)
        # concat
        outputs = torch.cat([title_output, attention_output], dim=-1)
        outputs = self.ln(outputs)
        sequence_output = self.dropout(outputs)
        logits = self.classifier(sequence_output)

        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(logits, labels), self.crf.decode(logits)
            return log_likelihood, sequence_of_tags, logits
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags, logits

