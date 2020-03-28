# coding: utf-8

import os
import torch
import joblib
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from gensim.models import KeyedVectors

import utils.json_util as ju
from utils.path_util import from_project_root, dirname

LABEL_IDS = {'neither': 0, 'CARDINAL': 1, 'CARDINALderiv': 2, 'CARDINALpart': 3, 'NAME': 4, 'NAMEderiv': 5, 'NAMEpart': 6, 'ORGCORP': 7, 'ORGCORPderiv': 8, 'ORGCORPpart': 9, 'UNIT': 10, 'UNITderiv': 11, 'UNITpart': 12, 'DATE': 13, 'DATEderiv': 14, 'DATEpart': 15, 'PER': 16, 'PERderiv': 17, 'PERpart': 18, 'DURATION': 19, 'DURATIONderiv': 20, 'DURATIONpart': 21, 'MONEY': 22, 'MONEYderiv': 23, 'MONEYpart': 24, 'MULT': 25, 'MULTderiv': 26, 'MULTpart': 27, 'FIRST': 28, 'FIRSTderiv': 29, 'FIRSTpart': 30, 'CITY': 31, 'CITYderiv': 32, 'CITYpart': 33, 'PERCENT': 34, 'PERCENTderiv': 35, 'PERCENTpart': 36, 'REL': 37, 'RELderiv': 38, 'RELpart': 39, 'HON': 40, 'HONderiv': 41, 'HONpart': 42, 'CORPJARGON': 43, 'CORPJARGONderiv': 44, 'CORPJARGONpart': 45, 'NATIONALITY': 46, 'NATIONALITYderiv': 47, 'NATIONALITYpart': 48, 'GOVERNMENT': 49, 'GOVERNMENTderiv': 50, 'GOVERNMENTpart': 51, 'COUNTRY': 52, 'COUNTRYderiv': 53, 'COUNTRYpart': 54, 'QUAL': 55, 'QUALderiv': 56, 'QUALpart': 57, 'MONTH': 58, 'MONTHderiv': 59, 'MONTHpart': 60, 'YEAR': 61, 'YEARderiv': 62, 'YEARpart': 63, 'STATE': 64, 'STATEderiv': 65, 'STATEpart': 66, 'ORDINAL': 67, 'ORDINALderiv': 68, 'ORDINALpart': 69, 'IPOINTS': 70, 'IPOINTSderiv': 71, 'IPOINTSpart': 72, 'ROLE': 73, 'ROLEderiv': 74, 'ROLEpart': 75, 'RATE': 76, 'RATEderiv': 77, 'RATEpart': 78, 'MEDIA': 79, 'MEDIAderiv': 80, 'MEDIApart': 81, 'NUMDAY': 82, 'NUMDAYderiv': 83, 'NUMDAYpart': 84, 'DAY': 85, 'DAYderiv': 86, 'DAYpart': 87, 'INI': 88, 'INIderiv': 89, 'INIpart': 90, 'NORPOTHER': 91, 'NORPOTHERderiv': 92, 'NORPOTHERpart': 93, 'ORGOTHER': 94, 'ORGOTHERderiv': 95, 'ORGOTHERpart': 96, 'PERIODIC': 97, 'PERIODICderiv': 98, 'PERIODICpart': 99, 'REGION': 100, 'REGIONderiv': 101, 'REGIONpart': 102, 'NORPPOLITICAL': 103, 'NORPPOLITICALderiv': 104, 'NORPPOLITICALpart': 105, 'AGE': 106, 'AGEderiv': 107, 'AGEpart': 108, 'INDEX': 109, 'INDEXderiv': 110, 'INDEXpart': 111, 'PRODUCTOTHER': 112, 'PRODUCTOTHERderiv': 113, 'PRODUCTOTHERpart': 114, 'STREET': 115, 'STREETderiv': 116, 'STREETpart': 117, 'QUANTITY0D': 118, 'QUANTITY0Dderiv': 119, 'QUANTITY0Dpart': 120, 'GRPORG': 121, 'GRPORGderiv': 122, 'GRPORGpart': 123, 'VEHICLE': 124, 'VEHICLEderiv': 125, 'VEHICLEpart': 126, 'ORGEDU': 127, 'ORGEDUderiv': 128, 'ORGEDUpart': 129, 'LAW': 130, 'LAWderiv': 131, 'LAWpart': 132, 'ORGPOLITICAL': 133, 'ORGPOLITICALderiv': 134, 'ORGPOLITICALpart': 135, 'BUILDING': 136, 'BUILDINGderiv': 137, 'BUILDINGpart': 138, 'CONTINENT': 139, 'CONTINENTderiv': 140, 'CONTINENTpart': 141, 'GPE': 142, 'GPEderiv': 143, 'GPEpart': 144, 'MIDDLE': 145, 'MIDDLEderiv': 146, 'MIDDLEpart': 147, 'SEASON': 148, 'SEASONderiv': 149, 'SEASONpart': 150, 'FOLD': 151, 'FOLDderiv': 152, 'FOLDpart': 153, 'OCEAN': 154, 'OCEANderiv': 155, 'OCEANpart': 156, 'WEIGHT': 157, 'WEIGHTderiv': 158, 'WEIGHTpart': 159, 'TIME': 160, 'TIMEderiv': 161, 'TIMEpart': 162, 'LOCATIONOTHER': 163, 'LOCATIONOTHERderiv': 164, 'LOCATIONOTHERpart': 165, 'DISEASE': 166, 'DISEASEderiv': 167, 'DISEASEpart': 168, 'EVENT': 169, 'EVENTderiv': 170, 'EVENTpart': 171, 'CITYSTATE': 172, 'CITYSTATEderiv': 173, 'CITYSTATEpart': 174, 'WOA': 175, 'WOAderiv': 176, 'WOApart': 177, 'SPORTSTEAM': 178, 'SPORTSTEAMderiv': 179, 'SPORTSTEAMpart': 180, 'DATEOTHER': 181, 'DATEOTHERderiv': 182, 'DATEOTHERpart': 183, 'GRPPER': 184, 'GRPPERderiv': 185, 'GRPPERpart': 186, 'NAMEMOD': 187, 'NAMEMODderiv': 188, 'NAMEMODpart': 189, 'BOOK': 190, 'BOOKderiv': 191, 'BOOKpart': 192, 'ELECTRONICS': 193, 'ELECTRONICSderiv': 194, 'ELECTRONICSpart': 195, 'ARMY': 196, 'ARMYderiv': 197, 'ARMYpart': 198, 'FACILITY': 199, 'FACILITYderiv': 200, 'FACILITYpart': 201, 'PRODUCTDRUG': 202, 'PRODUCTDRUGderiv': 203, 'PRODUCTDRUGpart': 204, 'TVSHOW': 205, 'TVSHOWderiv': 206, 'TVSHOWpart': 207, 'HURRICANE': 208, 'HURRICANEderiv': 209, 'HURRICANEpart': 210, 'SPORTSEVENT': 211, 'SPORTSEVENTderiv': 212, 'SPORTSEVENTpart': 213, 'NICKNAME': 214, 'NICKNAMEderiv': 215, 'NICKNAMEpart': 216, 'FILM': 217, 'FILMderiv': 218, 'FILMpart': 219, 'LANGUAGE': 220, 'LANGUAGEderiv': 221, 'LANGUAGEpart': 222, 'PRODUCTFOOD': 223, 'PRODUCTFOODderiv': 224, 'PRODUCTFOODpart': 225, 'RELIGION': 226, 'RELIGIONderiv': 227, 'RELIGIONpart': 228, 'SUBURB': 229, 'SUBURBderiv': 230, 'SUBURBpart': 231, 'GRPLOC': 232, 'GRPLOCderiv': 233, 'GRPLOCpart': 234, 'SONG': 235, 'SONGderiv': 236, 'SONGpart': 237, 'QUANTITYOTHER': 238, 'QUANTITYOTHERderiv': 239, 'QUANTITYOTHERpart': 240, 'SPACE': 241, 'SPACEderiv': 242, 'SPACEpart': 243, 'WAR': 244, 'WARderiv': 245, 'WARpart': 246, 'RIVER': 247, 'RIVERderiv': 248, 'RIVERpart': 249, 'CHEMICAL': 250, 'CHEMICALderiv': 251, 'CHEMICALpart': 252, 'FUND': 253, 'FUNDderiv': 254, 'FUNDpart': 255, 'BRIDGE': 256, 'BRIDGEderiv': 257, 'BRIDGEpart': 258, 'HOTEL': 259, 'HOTELderiv': 260, 'HOTELpart': 261, 'PLAY': 262, 'PLAYderiv': 263, 'PLAYpart': 264, 'STADIUM': 265, 'STADIUMderiv': 266, 'STADIUMpart': 267, 'AWARD': 268, 'AWARDderiv': 269, 'AWARDpart': 270, 'ORGRELIGIOUS': 271, 'ORGRELIGIOUSderiv': 272, 'ORGRELIGIOUSpart': 273, 'AIRPORT': 274, 'AIRPORTderiv': 275, 'AIRPORTpart': 276, 'GOD': 277, 'GODderiv': 278, 'GODpart': 279, 'ANIMATE': 280, 'ANIMATEderiv': 281, 'ANIMATEpart': 282, 'ATTRACTION': 283, 'ATTRACTIONderiv': 284, 'ATTRACTIONpart': 285, 'HOSPITAL': 286, 'HOSPITALderiv': 287, 'HOSPITALpart': 288, 'WEAPON': 289, 'WEAPONderiv': 290, 'WEAPONpart': 291, 'MUSEUM': 292, 'MUSEUMderiv': 293, 'MUSEUMpart': 294, 'ENERGY': 295, 'ENERGYderiv': 296, 'ENERGYpart': 297, 'PAINTING': 298, 'PAINTINGderiv': 299, 'PAINTINGpart': 300, 'SPEED': 301, 'SPEEDderiv': 302, 'SPEEDpart': 303, 'BAND': 304, 'BANDderiv': 305, 'BANDpart': 306, 'SPORTSSEASON': 307, 'SPORTSSEASONderiv': 308, 'SPORTSSEASONpart': 309, 'SCINAME': 310, 'SCINAMEderiv': 311, 'SCINAMEpart': 312, 'ADDRESSNON': 313, 'ADDRESSNONderiv': 314, 'ADDRESSNONpart': 315, 'ALBUM': 316, 'ALBUMderiv': 317, 'ALBUMpart': 318, 'CONCERT': 319, 'CONCERTderiv': 320, 'CONCERTpart': 321, 'NATURALDISASTER': 322, 'NATURALDISASTERderiv': 323, 'NATURALDISASTERpart': 324, 'TEMPERATURE': 325, 'TEMPERATUREderiv': 326, 'TEMPERATUREpart': 327, 'BORDER': 328, 'BORDERderiv': 329, 'BORDERpart': 330, 'CHANNEL': 331, 'CHANNELderiv': 332, 'CHANNELpart': 333, 'STATION': 334, 'STATIONderiv': 335, 'STATIONpart': 336}
PRETRAINED_URL = from_project_root("data/embedding/PubMed-shuffle-win-30.bin")
LABEL_LIST = {'O', 'CARDINAL', 'CARDINALderiv', 'CARDINALpart', 'NAME', 'NAMEderiv', 'NAMEpart', 'ORGCORP', 'ORGCORPderiv', 'ORGCORPpart', 'UNIT', 'UNITderiv', 'UNITpart', 'DATE', 'DATEderiv', 'DATEpart', 'PER', 'PERderiv', 'PERpart', 'DURATION', 'DURATIONderiv', 'DURATIONpart', 'MONEY', 'MONEYderiv', 'MONEYpart', 'MULT', 'MULTderiv', 'MULTpart', 'FIRST', 'FIRSTderiv', 'FIRSTpart', 'CITY', 'CITYderiv', 'CITYpart', 'PERCENT', 'PERCENTderiv', 'PERCENTpart', 'REL', 'RELderiv', 'RELpart', 'HON', 'HONderiv', 'HONpart', 'CORPJARGON', 'CORPJARGONderiv', 'CORPJARGONpart', 'NATIONALITY', 'NATIONALITYderiv', 'NATIONALITYpart', 'GOVERNMENT', 'GOVERNMENTderiv', 'GOVERNMENTpart', 'COUNTRY', 'COUNTRYderiv', 'COUNTRYpart', 'QUAL', 'QUALderiv', 'QUALpart', 'MONTH', 'MONTHderiv', 'MONTHpart', 'YEAR', 'YEARderiv', 'YEARpart', 'STATE', 'STATEderiv', 'STATEpart', 'ORDINAL', 'ORDINALderiv', 'ORDINALpart', 'IPOINTS', 'IPOINTSderiv', 'IPOINTSpart', 'ROLE', 'ROLEderiv', 'ROLEpart', 'RATE', 'RATEderiv', 'RATEpart', 'MEDIA', 'MEDIAderiv', 'MEDIApart', 'NUMDAY', 'NUMDAYderiv', 'NUMDAYpart', 'DAY', 'DAYderiv', 'DAYpart', 'INI', 'INIderiv', 'INIpart', 'NORPOTHER', 'NORPOTHERderiv', 'NORPOTHERpart', 'ORGOTHER', 'ORGOTHERderiv', 'ORGOTHERpart', 'PERIODIC', 'PERIODICderiv', 'PERIODICpart', 'REGION', 'REGIONderiv', 'REGIONpart', 'NORPPOLITICAL', 'NORPPOLITICALderiv', 'NORPPOLITICALpart', 'AGE', 'AGEderiv', 'AGEpart', 'INDEX', 'INDEXderiv', 'INDEXpart', 'PRODUCTOTHER', 'PRODUCTOTHERderiv', 'PRODUCTOTHERpart', 'STREET', 'STREETderiv', 'STREETpart', 'QUANTITY0D', 'QUANTITY0Dderiv', 'QUANTITY0Dpart', 'GRPORG', 'GRPORGderiv', 'GRPORGpart', 'VEHICLE', 'VEHICLEderiv', 'VEHICLEpart', 'ORGEDU', 'ORGEDUderiv', 'ORGEDUpart', 'LAW', 'LAWderiv', 'LAWpart', 'ORGPOLITICAL', 'ORGPOLITICALderiv', 'ORGPOLITICALpart', 'BUILDING', 'BUILDINGderiv', 'BUILDINGpart', 'CONTINENT', 'CONTINENTderiv', 'CONTINENTpart', 'GPE', 'GPEderiv', 'GPEpart', 'MIDDLE', 'MIDDLEderiv', 'MIDDLEpart', 'SEASON', 'SEASONderiv', 'SEASONpart', 'FOLD', 'FOLDderiv', 'FOLDpart', 'OCEAN', 'OCEANderiv', 'OCEANpart', 'WEIGHT', 'WEIGHTderiv', 'WEIGHTpart', 'TIME', 'TIMEderiv', 'TIMEpart', 'LOCATIONOTHER', 'LOCATIONOTHERderiv', 'LOCATIONOTHERpart', 'DISEASE', 'DISEASEderiv', 'DISEASEpart', 'EVENT', 'EVENTderiv', 'EVENTpart', 'CITYSTATE', 'CITYSTATEderiv', 'CITYSTATEpart', 'WOA', 'WOAderiv', 'WOApart', 'SPORTSTEAM', 'SPORTSTEAMderiv', 'SPORTSTEAMpart', 'DATEOTHER', 'DATEOTHERderiv', 'DATEOTHERpart', 'GRPPER', 'GRPPERderiv', 'GRPPERpart', 'NAMEMOD', 'NAMEMODderiv', 'NAMEMODpart', 'BOOK', 'BOOKderiv', 'BOOKpart', 'ELECTRONICS', 'ELECTRONICSderiv', 'ELECTRONICSpart', 'ARMY', 'ARMYderiv', 'ARMYpart', 'FACILITY', 'FACILITYderiv', 'FACILITYpart', 'PRODUCTDRUG', 'PRODUCTDRUGderiv', 'PRODUCTDRUGpart', 'TVSHOW', 'TVSHOWderiv', 'TVSHOWpart', 'HURRICANE', 'HURRICANEderiv', 'HURRICANEpart', 'SPORTSEVENT', 'SPORTSEVENTderiv', 'SPORTSEVENTpart', 'NICKNAME', 'NICKNAMEderiv', 'NICKNAMEpart', 'FILM', 'FILMderiv', 'FILMpart', 'LANGUAGE', 'LANGUAGEderiv', 'LANGUAGEpart', 'PRODUCTFOOD', 'PRODUCTFOODderiv', 'PRODUCTFOODpart', 'RELIGION', 'RELIGIONderiv', 'RELIGIONpart', 'SUBURB', 'SUBURBderiv', 'SUBURBpart', 'GRPLOC', 'GRPLOCderiv', 'GRPLOCpart', 'SONG', 'SONGderiv', 'SONGpart', 'QUANTITYOTHER', 'QUANTITYOTHERderiv', 'QUANTITYOTHERpart', 'SPACE', 'SPACEderiv', 'SPACEpart', 'WAR', 'WARderiv', 'WARpart', 'RIVER', 'RIVERderiv', 'RIVERpart', 'CHEMICAL', 'CHEMICALderiv', 'CHEMICALpart', 'FUND', 'FUNDderiv', 'FUNDpart', 'BRIDGE', 'BRIDGEderiv', 'BRIDGEpart', 'HOTEL', 'HOTELderiv', 'HOTELpart', 'PLAY', 'PLAYderiv', 'PLAYpart', 'STADIUM', 'STADIUMderiv', 'STADIUMpart', 'AWARD', 'AWARDderiv', 'AWARDpart', 'ORGRELIGIOUS', 'ORGRELIGIOUSderiv', 'ORGRELIGIOUSpart', 'AIRPORT', 'AIRPORTderiv', 'AIRPORTpart', 'GOD', 'GODderiv', 'GODpart', 'ANIMATE', 'ANIMATEderiv', 'ANIMATEpart', 'ATTRACTION', 'ATTRACTIONderiv', 'ATTRACTIONpart', 'HOSPITAL', 'HOSPITALderiv', 'HOSPITALpart', 'WEAPON', 'WEAPONderiv', 'WEAPONpart', 'MUSEUM', 'MUSEUMderiv', 'MUSEUMpart', 'ENERGY', 'ENERGYderiv', 'ENERGYpart', 'PAINTING', 'PAINTINGderiv', 'PAINTINGpart', 'SPEED', 'SPEEDderiv', 'SPEEDpart', 'BAND', 'BANDderiv', 'BANDpart', 'SPORTSSEASON', 'SPORTSSEASONderiv', 'SPORTSSEASONpart', 'SCINAME', 'SCINAMEderiv', 'SCINAMEpart', 'ADDRESSNON', 'ADDRESSNONderiv', 'ADDRESSNONpart', 'ALBUM', 'ALBUMderiv', 'ALBUMpart', 'CONCERT', 'CONCERTderiv', 'CONCERTpart', 'NATURALDISASTER', 'NATURALDISASTERderiv', 'NATURALDISASTERpart', 'TEMPERATURE', 'TEMPERATUREderiv', 'TEMPERATUREpart', 'BORDER', 'BORDERderiv', 'BORDERpart', 'CHANNEL', 'CHANNELderiv', 'CHANNELpart', 'STATION', 'STATIONderiv', 'STATIONpart'}


class End2EndDataset(Dataset):
    def __init__(self, data_url, device, evaluating=False):
        super().__init__()
        self.data_url = data_url
        self.label_ids = LABEL_IDS
        self.label_list = LABEL_LIST
        self.sentences, self.records = load_raw_data(data_url)
        self.device = device
        self.evaluating = evaluating

    def __getitem__(self, index):
        return self.sentences[index], self.records[index]

    def __len__(self):
        return len(self.sentences)

    def collate_func(self, data_list):
        data_list = sorted(data_list, key=lambda tup: len(tup[0]), reverse=True)
        sentence_list, records_list = zip(*data_list)  # un zip
        sentence_tensors = gen_sentence_tensors(sentence_list, self.device, self.data_url)
        # (sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices)

        max_sent_len = sentence_tensors[1][0]
        sentence_labels = list()
        region_labels = list()
        for records, length in zip(records_list, sentence_tensors[1]):
            labels = [0] * max_sent_len
        #    print(records)
            for record in records:
                for i in range(record[0]+1,record[1]-1):
                    if labels[i] == 1 or labels[i] == 2:
                        continue
                    labels[i] = 3
                labels[record[1]-1] = 2
                labels[record[0]] = 1

            sentence_labels.append(labels)

            for start in range(0, length):
                if labels[start] == 1:
                    region_labels.append(self.label_ids[records[(start, start+1)]] if (start, start+1) in records else 0)
                    for end in range(start+1, length):
                        if labels[end] == 2:
                            region_labels.append(self.label_ids[records[(start, end+1)]] if (start, end+1) in records else 0)


        sentence_labels = torch.LongTensor(sentence_labels).to(self.device)
        region_labels = torch.LongTensor(region_labels).to(self.device)

        if self.evaluating:
            return sentence_tensors, sentence_labels, region_labels, records_list
        return sentence_tensors, sentence_labels, region_labels


def gen_sentence_tensors(sentence_list, device, data_url):
    """ generate input tensors from sentence list

    Args:
        sentence_list: list of raw sentence
        device: torch device
        data_url: data_url used to locate vocab files

    Returns:
        sentences, tensor
        sentence_lengths, tensor
        sentence_words, list of tensor
        sentence_word_lengths, list of tensor
        sentence_word_indices, list of tensor

    """
    vocab = ju.load(dirname(data_url) + '/vocab.json')
    char_vocab = ju.load(dirname(data_url) + '/char_vocab.json')

    sentences = list()
    sentence_words = list()
    sentence_word_lengths = list()
    sentence_word_indices = list()

    unk_idx = 1
    for sent in sentence_list:
        # word to word id
        sentence = torch.LongTensor([vocab[word] if word in vocab else unk_idx
                                     for word in sent]).to(device)

        # char of word to char id
        words = list()
        for word in sent:
            words.append([char_vocab[ch] if ch in char_vocab else unk_idx
                          for ch in word])

        # save word lengths
        word_lengths = torch.LongTensor([len(word) for word in words]).to(device)

        # sorting lengths according to length
        word_lengths, word_indices = torch.sort(word_lengths, descending=True)

        # sorting word according word length
        words = np.array(words)[word_indices.cpu().numpy()]
        word_indices = word_indices.to(device)
        words = [torch.LongTensor(word).to(device) for word in words]

        # padding char tensor of words
        words = pad_sequence(words, batch_first=True).to(device)
        # (max_word_len, sent_len)

        sentences.append(sentence)
        sentence_words.append(words)
        sentence_word_lengths.append(word_lengths)
        sentence_word_indices.append(word_indices)

    # record sentence length and padding sentences
    sentence_lengths = [len(sentence) for sentence in sentences]
    # (batch_size)
    sentences = pad_sequence(sentences, batch_first=True).to()
    # (batch_size, max_sent_len)

    return sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices


def gen_vocab_from_data(data_urls, pretrained_url, binary=True, update=False, min_count=1):
    """ generate vocabulary and embeddings from data file, generated vocab files will be saved in
        data dir

    Args:
        data_urls: url to data file(s), list or string
        pretrained_url: url to pretrained embedding file
        binary: binary for load word2vec
        update: force to update even vocab file exists
        min_count: minimum count of a word

    Returns:
        generated word embedding url
    """

    if isinstance(data_urls, str):
        data_urls = [data_urls]
    data_dir = os.path.dirname(data_urls[0])
    vocab_url = os.path.join(data_dir, "vocab.json")
    char_vocab_url = os.path.join(data_dir, "char_vocab.json")
    embedding_url = os.path.join(data_dir, "embeddings.npy") if pretrained_url else None

    if (not update) and os.path.exists(vocab_url):
        print("vocab file already exists")
        return embedding_url

    vocab = set()
    char_vocab = set()
    word_counts = defaultdict(int)
    print("generating vocab from", data_urls)
    for data_url in data_urls:
        with open(data_url, 'r', encoding='utf-8') as data_file:
            for row in data_file:
                if row == '\n':
                    continue
                token = row.split()[0]
                word_counts[token] += 1
                if word_counts[token] > min_count:
                    vocab.add(row.split()[0])
                char_vocab = char_vocab.union(row.split()[0])

    # sorting vocab according alphabet order
    vocab = sorted(vocab)
    char_vocab = sorted(char_vocab)

    # generate word embeddings for vocab
    if pretrained_url is not None:
        print("generating pre-trained embedding from", pretrained_url)
        kvs = KeyedVectors.load_word2vec_format(pretrained_url, binary=binary)
        embeddings = list()
        for word in vocab:
            if word in kvs:
                embeddings.append(kvs[word])
            else:
                embeddings.append(np.random.uniform(-0.25, 0.25, kvs.vector_size)),

    char_vocab = ['<pad', '<unk>'] + char_vocab
    vocab = ['<pad>', '<unk>'] + vocab
    ju.dump(ju.list_to_dict(vocab), vocab_url)
    ju.dump(ju.list_to_dict(char_vocab), char_vocab_url)

    if pretrained_url is None:
        return
    embeddings = np.vstack([np.zeros(kvs.vector_size),  # for <pad>
                            np.random.uniform(-0.25, 0.25, kvs.vector_size),  # for <unk>
                            embeddings])
    np.save(embedding_url, embeddings)
    return embedding_url


def infer_records(columns):
    """ inferring all entity records of a sentence

    Args:
        columns: columns of a sentence in iob2 format

    Returns:
        entity record in gave sentence

    """
    records = dict()
    for col in columns:
        start = 0
        while start < len(col):
            end = start + 1
            if col[start][0] == 'B':
                while end < len(col) and col[end][0] == 'I':
                    end += 1
                records[(start, end)] = col[start][2:]
            start = end
    return records


def load_raw_data(data_url, update=False):
    """ load data into sentences and records

    Args:
        data_url: url to data file
        update: whether force to update
    Returns:
        sentences(raw), records
    """

    # load from pickle
    save_url = data_url.replace('.bio', '.raw.pkl').replace('.iob2', '.raw.pkl')
    if not update and os.path.exists(save_url):
        return joblib.load(save_url)

    sentences = list()
    records = list()
    with open(data_url, 'r', encoding='utf-8') as iob_file:
        first_line = iob_file.readline()
        n_columns = first_line.count('\t')
        # JNLPBA dataset don't contains the extra 'O' column
        if 'jnlpba' in data_url:
            n_columns += 1
        columns = [[x] for x in first_line.split()]
        for line in iob_file:
            if line != '\n':
                line_values = line.split()
                for i in range(n_columns):
                    columns[i].append(line_values[i])

            else:  # end of a sentence
                sentence = columns[0]
                sentences.append(sentence)
                records.append(infer_records(columns[1:]))
                columns = [list() for i in range(n_columns)]
    joblib.dump((sentences, records), save_url)
    return sentences, records


def prepare_vocab(data_url, pretrained_url=PRETRAINED_URL, update=True, min_count=0):
    """ prepare vocab and embedding

    Args:
        data_url: url to data file for preparing vocab
        pretrained_url: url to pre-trained embedding file
        update: force to update
        min_count: minimum count for gen_vocab

    """
    binary = pretrained_url.endswith('.bin')
    gen_vocab_from_data(data_url, pretrained_url, binary=binary, update=update, min_count=min_count)


def main():
    # load_data(data_url, update=False)
    data_urls = [from_project_root("data/Germ/germ.train.iob2"),
                 from_project_root("data/Germ/germ.dev.iob2"),
                 from_project_root("data/Germ/germ.test.iob2")]
    prepare_vocab(data_urls, update=True, min_count=1)
    pass


if __name__ == '__main__':
    main()
