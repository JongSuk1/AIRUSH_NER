import pandas as pd
import torch


class MenuOptionDataset(object):
    def __init__(self, tokenizer, dataset_file_path):
        self.tokenizer = tokenizer
        self.total_data = pd.read_json(dataset_file_path)
        self.label_to_idx = {label: i for i, label in enumerate(
            ["X", "B", "I", 'S', 'O', "[CLS]", "[SEP]"])}
        self.MAX_LEN = 80
        self.MAX_ATTR_LEN = 10

    def subfinder(self, mylist, pattern):
        return filter(set(pattern).__contains__, mylist)

    def __getitem__(self, idx):
        menu_text = self.total_data.iloc[idx]['text']
        entity = self.total_data.iloc[idx]['entity']
        value = self.total_data.iloc[idx]['value']

        assert value in menu_text
        s = self.total_data.iloc[idx]['value_start']
        e = self.total_data.iloc[idx]['value_end']
        offset_mapping = self.tokenizer.encode_plus(menu_text, return_offsets_mapping=True, add_special_tokens=False)[
            'offset_mapping']
        m_tokens = self.tokenizer.tokenize(menu_text)
        labels = ['O'] * len(m_tokens)
        is_tag_b = False

        for idx, offset in enumerate(offset_mapping):
            if offset[0] <= s and e < offset[1] and not is_tag_b:
                labels[idx] = 'S'
                break
            elif offset[0] <= s and s < offset[1] and not is_tag_b:
                labels[idx] = 'B'
                is_tag_b = True
            elif offset[0] <= e and offset[0] > s:
                labels[idx] = 'I'

        e_tokens = self.tokenizer.tokenize(entity)
        label_ids = [self.label_to_idx[x] for x in labels]
        special_tokens_count = 2
        if len(m_tokens) > self.MAX_LEN - special_tokens_count:
            m_tokens = m_tokens[: (self.MAX_LEN - special_tokens_count)]
            label_ids = label_ids[: (self.MAX_LEN - special_tokens_count)]
        if len(e_tokens) > self.MAX_ATTR_LEN - special_tokens_count:
            e_tokens = e_tokens[: (self.MAX_ATTR_LEN - special_tokens_count)]

        m_tokens += ['[SEP]']
        e_tokens += ['[SEP]']
        label_ids += [self.label_to_idx['[SEP]']]
        m_segment_ids = [0] * len(m_tokens)
        e_segment_ids = [0] * len(e_tokens)

        cls_token = '[CLS]'
        m_tokens = [cls_token] + m_tokens
        label_ids = [self.label_to_idx[cls_token]] + label_ids
        m_segment_ids = [1] + m_segment_ids
        e_tokens = [cls_token] + e_tokens
        e_segment_ids = [1] + e_segment_ids
        m_input_ids = self.tokenizer.convert_tokens_to_ids(m_tokens)
        e_input_ids = self.tokenizer.convert_tokens_to_ids(e_tokens)

        m_input_mask = [1] * len(m_input_ids)
        e_input_mask = [1] * len(e_input_ids)

        r_padding_length = self.MAX_LEN - len(m_input_ids)
        a_padding_length = self.MAX_ATTR_LEN - len(e_input_ids)
        m_input_ids += [0] * r_padding_length
        m_input_mask += [0] * r_padding_length

        label_ids += [0] * r_padding_length

        e_input_ids += [0] * a_padding_length
        e_input_mask += [0] * a_padding_length

        assert len(m_input_ids) == self.MAX_LEN
        assert len(m_input_mask) == self.MAX_LEN
        assert len(label_ids) == self.MAX_LEN
        assert len(e_input_ids) == self.MAX_ATTR_LEN
        assert len(e_input_mask) == self.MAX_ATTR_LEN

        target = {}
        target['m_input_ids'] = torch.as_tensor(m_input_ids)
        target['m_input_mask'] = torch.as_tensor(m_input_mask)
        target['e_input_ids'] = torch.as_tensor(e_input_ids)
        target['e_input_mask'] = torch.as_tensor(e_input_mask)
        target['label_ids'] = torch.as_tensor(label_ids)

        return target

    def get_label_to_idx(self):
        return self.label_to_idx

    def __len__(self):
        return len(self.total_data)
