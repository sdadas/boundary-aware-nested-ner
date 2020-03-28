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

LABEL_IDS = {'neither': 0, 'CARDINAL': 1, 'CARDINALderiv': 2, 'CARDINALpart': 3, 'NAME': 4, 'NAMEderiv': 5, 'NAMEpart': 6, 'ORGCORP': 7, 'ORGCORPderiv': 8, 'ORGCORPpart': 9, 'UNIT': 10, 'UNITderiv': 11, 'UNITpart': 12, 'DATE': 13, 'DATEderiv': 14, 'DATEpart': 15, 'PER': 16, 'PERderiv': 17, 'PERpart': 18, 'DURATION': 19, 'DURATIONderiv': 20, 'DURATIONpart': 21, 'MONEY': 22, 'MONEYderiv': 23, 'MONEYpart': 24, 'MULT': 25, 'MULTderiv': 26, 'MULTpart': 27, 'FIRST': 28, 'FIRSTderiv': 29, 'FIRSTpart': 30, 'CITY': 31, 'CITYderiv': 32, 'CITYpart': 33, 'PERCENT': 34, 'PERCENTderiv': 35, 'PERCENTpart': 36, 'REL': 37, 'RELderiv': 38, 'RELpart': 39, 'HON': 40, 'HONderiv': 41, 'HONpart': 42, 'CORPJARGON': 43, 'CORPJARGONderiv': 44, 'CORPJARGONpart': 45, 'NATIONALITY': 46, 'NATIONALITYderiv': 47, 'NATIONALITYpart': 48, 'GOVERNMENT': 49, 'GOVERNMENTderiv': 50, 'GOVERNMENTpart': 51, 'COUNTRY': 52, 'COUNTRYderiv': 53, 'COUNTRYpart': 54, 'QUAL': 55, 'QUALderiv': 56, 'QUALpart': 57, 'MONTH': 58, 'MONTHderiv': 59, 'MONTHpart': 60, 'YEAR': 61, 'YEARderiv': 62, 'YEARpart': 63, 'STATE': 64, 'STATEderiv': 65, 'STATEpart': 66, 'ORDINAL': 67, 'ORDINALderiv': 68, 'ORDINALpart': 69, 'IPOINTS': 70, 'IPOINTSderiv': 71, 'IPOINTSpart': 72, 'ROLE': 73, 'ROLEderiv': 74, 'ROLEpart': 75, 'RATE': 76, 'RATEderiv': 77, 'RATEpart': 78, 'MEDIA': 79, 'MEDIAderiv': 80, 'MEDIApart': 81, 'NUMDAY': 82, 'NUMDAYderiv': 83, 'NUMDAYpart': 84, 'DAY': 85, 'DAYderiv': 86, 'DAYpart': 87, 'INI': 88, 'INIderiv': 89, 'INIpart': 90, 'NORPOTHER': 91, 'NORPOTHERderiv': 92, 'NORPOTHERpart': 93, 'ORGOTHER': 94, 'ORGOTHERderiv': 95, 'ORGOTHERpart': 96, 'PERIODIC': 97, 'PERIODICderiv': 98, 'PERIODICpart': 99, 'REGION': 100, 'REGIONderiv': 101, 'REGIONpart': 102, 'NORPPOLITICAL': 103, 'NORPPOLITICALderiv': 104, 'NORPPOLITICALpart': 105, 'AGE': 106, 'AGEderiv': 107, 'AGEpart': 108, 'INDEX': 109, 'INDEXderiv': 110, 'INDEXpart': 111, 'PRODUCTOTHER': 112, 'PRODUCTOTHERderiv': 113, 'PRODUCTOTHERpart': 114, 'STREET': 115, 'STREETderiv': 116, 'STREETpart': 117, 'QUANTITY0D': 118, 'QUANTITY0Dderiv': 119, 'QUANTITY0Dpart': 120, 'QUANTITY1D': 121, 'QUANTITY1Dderiv': 122, 'QUANTITY1Dpart': 123, 'QUANTITY2D': 124, 'QUANTITY2Dderiv': 125, 'QUANTITY2Dpart': 126, 'QUANTITY3D': 127, 'QUANTITY3Dderiv': 128, 'QUANTITY3Dpart': 129, 'GRPORG': 130, 'GRPORGderiv': 131, 'GRPORGpart': 132, 'VEHICLE': 133, 'VEHICLEderiv': 134, 'VEHICLEpart': 135, 'ORGEDU': 136, 'ORGEDUderiv': 137, 'ORGEDUpart': 138, 'LAW': 139, 'LAWderiv': 140, 'LAWpart': 141, 'ORGPOLITICAL': 142, 'ORGPOLITICALderiv': 143, 'ORGPOLITICALpart': 144, 'BUILDING': 145, 'BUILDINGderiv': 146, 'BUILDINGpart': 147, 'CONTINENT': 148, 'CONTINENTderiv': 149, 'CONTINENTpart': 150, 'GPE': 151, 'GPEderiv': 152, 'GPEpart': 153, 'MIDDLE': 154, 'MIDDLEderiv': 155, 'MIDDLEpart': 156, 'SEASON': 157, 'SEASONderiv': 158, 'SEASONpart': 159, 'FOLD': 160, 'FOLDderiv': 161, 'FOLDpart': 162, 'OCEAN': 163, 'OCEANderiv': 164, 'OCEANpart': 165, 'WEIGHT': 166, 'WEIGHTderiv': 167, 'WEIGHTpart': 168, 'TIME': 169, 'TIMEderiv': 170, 'TIMEpart': 171, 'LOCATIONOTHER': 172, 'LOCATIONOTHERderiv': 173, 'LOCATIONOTHERpart': 174, 'DISEASE': 175, 'DISEASEderiv': 176, 'DISEASEpart': 177, 'EVENT': 178, 'EVENTderiv': 179, 'EVENTpart': 180, 'CITYSTATE': 181, 'CITYSTATEderiv': 182, 'CITYSTATEpart': 183, 'WOA': 184, 'WOAderiv': 185, 'WOApart': 186, 'SPORTSTEAM': 187, 'SPORTSTEAMderiv': 188, 'SPORTSTEAMpart': 189, 'DATEOTHER': 190, 'DATEOTHERderiv': 191, 'DATEOTHERpart': 192, 'GRPPER': 193, 'GRPPERderiv': 194, 'GRPPERpart': 195, 'NAMEMOD': 196, 'NAMEMODderiv': 197, 'NAMEMODpart': 198, 'BOOK': 199, 'BOOKderiv': 200, 'BOOKpart': 201, 'ELECTRONICS': 202, 'ELECTRONICSderiv': 203, 'ELECTRONICSpart': 204, 'ARMY': 205, 'ARMYderiv': 206, 'ARMYpart': 207, 'FACILITY': 208, 'FACILITYderiv': 209, 'FACILITYpart': 210, 'PRODUCTDRUG': 211, 'PRODUCTDRUGderiv': 212, 'PRODUCTDRUGpart': 213, 'TVSHOW': 214, 'TVSHOWderiv': 215, 'TVSHOWpart': 216, 'HURRICANE': 217, 'HURRICANEderiv': 218, 'HURRICANEpart': 219, 'SPORTSEVENT': 220, 'SPORTSEVENTderiv': 221, 'SPORTSEVENTpart': 222, 'NICKNAME': 223, 'NICKNAMEderiv': 224, 'NICKNAMEpart': 225, 'FILM': 226, 'FILMderiv': 227, 'FILMpart': 228, 'LANGUAGE': 229, 'LANGUAGEderiv': 230, 'LANGUAGEpart': 231, 'PRODUCTFOOD': 232, 'PRODUCTFOODderiv': 233, 'PRODUCTFOODpart': 234, 'RELIGION': 235, 'RELIGIONderiv': 236, 'RELIGIONpart': 237, 'SUBURB': 238, 'SUBURBderiv': 239, 'SUBURBpart': 240, 'GRPLOC': 241, 'GRPLOCderiv': 242, 'GRPLOCpart': 243, 'SONG': 244, 'SONGderiv': 245, 'SONGpart': 246, 'QUANTITYOTHER': 247, 'QUANTITYOTHERderiv': 248, 'QUANTITYOTHERpart': 249, 'SPACE': 250, 'SPACEderiv': 251, 'SPACEpart': 252, 'WAR': 253, 'WARderiv': 254, 'WARpart': 255, 'RIVER': 256, 'RIVERderiv': 257, 'RIVERpart': 258, 'CHEMICAL': 259, 'CHEMICALderiv': 260, 'CHEMICALpart': 261, 'FUND': 262, 'FUNDderiv': 263, 'FUNDpart': 264, 'BRIDGE': 265, 'BRIDGEderiv': 266, 'BRIDGEpart': 267, 'HOTEL': 268, 'HOTELderiv': 269, 'HOTELpart': 270, 'PLAY': 271, 'PLAYderiv': 272, 'PLAYpart': 273, 'STADIUM': 274, 'STADIUMderiv': 275, 'STADIUMpart': 276, 'AWARD': 277, 'AWARDderiv': 278, 'AWARDpart': 279, 'ORGRELIGIOUS': 280, 'ORGRELIGIOUSderiv': 281, 'ORGRELIGIOUSpart': 282, 'AIRPORT': 283, 'AIRPORTderiv': 284, 'AIRPORTpart': 285, 'GOD': 286, 'GODderiv': 287, 'GODpart': 288, 'ANIMATE': 289, 'ANIMATEderiv': 290, 'ANIMATEpart': 291, 'ATTRACTION': 292, 'ATTRACTIONderiv': 293, 'ATTRACTIONpart': 294, 'HOSPITAL': 295, 'HOSPITALderiv': 296, 'HOSPITALpart': 297, 'WEAPON': 298, 'WEAPONderiv': 299, 'WEAPONpart': 300, 'MUSEUM': 301, 'MUSEUMderiv': 302, 'MUSEUMpart': 303, 'ENERGY': 304, 'ENERGYderiv': 305, 'ENERGYpart': 306, 'PAINTING': 307, 'PAINTINGderiv': 308, 'PAINTINGpart': 309, 'SPEED': 310, 'SPEEDderiv': 311, 'SPEEDpart': 312, 'BAND': 313, 'BANDderiv': 314, 'BANDpart': 315, 'SPORTSSEASON': 316, 'SPORTSSEASONderiv': 317, 'SPORTSSEASONpart': 318, 'SCINAME': 319, 'SCINAMEderiv': 320, 'SCINAMEpart': 321, 'ADDRESSNON': 322, 'ADDRESSNONderiv': 323, 'ADDRESSNONpart': 324, 'ALBUM': 325, 'ALBUMderiv': 326, 'ALBUMpart': 327, 'CONCERT': 328, 'CONCERTderiv': 329, 'CONCERTpart': 330, 'NATURALDISASTER': 331, 'NATURALDISASTERderiv': 332, 'NATURALDISASTERpart': 333, 'TEMPERATURE': 334, 'TEMPERATUREderiv': 335, 'TEMPERATUREpart': 336, 'BORDER': 337, 'BORDERderiv': 338, 'BORDERpart': 339, 'CHANNEL': 340, 'CHANNELderiv': 341, 'CHANNELpart': 342, 'STATION': 343, 'STATIONderiv': 344, 'STATIONpart': 345}
PRETRAINED_URL = from_project_root("data/embedding/PubMed-shuffle-win-30.bin")
LABEL_LIST = {'O', 'CARDINAL', 'CARDINALderiv', 'CARDINALpart', 'NAME', 'NAMEderiv', 'NAMEpart', 'ORGCORP', 'ORGCORPderiv', 'ORGCORPpart', 'UNIT', 'UNITderiv', 'UNITpart', 'DATE', 'DATEderiv', 'DATEpart', 'PER', 'PERderiv', 'PERpart', 'DURATION', 'DURATIONderiv', 'DURATIONpart', 'MONEY', 'MONEYderiv', 'MONEYpart', 'MULT', 'MULTderiv', 'MULTpart', 'FIRST', 'FIRSTderiv', 'FIRSTpart', 'CITY', 'CITYderiv', 'CITYpart', 'PERCENT', 'PERCENTderiv', 'PERCENTpart', 'REL', 'RELderiv', 'RELpart', 'HON', 'HONderiv', 'HONpart', 'CORPJARGON', 'CORPJARGONderiv', 'CORPJARGONpart', 'NATIONALITY', 'NATIONALITYderiv', 'NATIONALITYpart', 'GOVERNMENT', 'GOVERNMENTderiv', 'GOVERNMENTpart', 'COUNTRY', 'COUNTRYderiv', 'COUNTRYpart', 'QUAL', 'QUALderiv', 'QUALpart', 'MONTH', 'MONTHderiv', 'MONTHpart', 'YEAR', 'YEARderiv', 'YEARpart', 'STATE', 'STATEderiv', 'STATEpart', 'ORDINAL', 'ORDINALderiv', 'ORDINALpart', 'IPOINTS', 'IPOINTSderiv', 'IPOINTSpart', 'ROLE', 'ROLEderiv', 'ROLEpart', 'RATE', 'RATEderiv', 'RATEpart', 'MEDIA', 'MEDIAderiv', 'MEDIApart', 'NUMDAY', 'NUMDAYderiv', 'NUMDAYpart', 'DAY', 'DAYderiv', 'DAYpart', 'INI', 'INIderiv', 'INIpart', 'NORPOTHER', 'NORPOTHERderiv', 'NORPOTHERpart', 'ORGOTHER', 'ORGOTHERderiv', 'ORGOTHERpart', 'PERIODIC', 'PERIODICderiv', 'PERIODICpart', 'REGION', 'REGIONderiv', 'REGIONpart', 'NORPPOLITICAL', 'NORPPOLITICALderiv', 'NORPPOLITICALpart', 'AGE', 'AGEderiv', 'AGEpart', 'INDEX', 'INDEXderiv', 'INDEXpart', 'PRODUCTOTHER', 'PRODUCTOTHERderiv', 'PRODUCTOTHERpart', 'STREET', 'STREETderiv', 'STREETpart', 'QUANTITY0D', 'QUANTITY0Dderiv', 'QUANTITY0Dpart', 'QUANTITY1D', 'QUANTITY1Dderiv', 'QUANTITY1Dpart', 'QUANTITY2D', 'QUANTITY2Dderiv', 'QUANTITY2Dpart', 'QUANTITY3D', 'QUANTITY3Dderiv', 'QUANTITY3Dpart', 'GRPORG', 'GRPORGderiv', 'GRPORGpart', 'VEHICLE', 'VEHICLEderiv', 'VEHICLEpart', 'ORGEDU', 'ORGEDUderiv', 'ORGEDUpart', 'LAW', 'LAWderiv', 'LAWpart', 'ORGPOLITICAL', 'ORGPOLITICALderiv', 'ORGPOLITICALpart', 'BUILDING', 'BUILDINGderiv', 'BUILDINGpart', 'CONTINENT', 'CONTINENTderiv', 'CONTINENTpart', 'GPE', 'GPEderiv', 'GPEpart', 'MIDDLE', 'MIDDLEderiv', 'MIDDLEpart', 'SEASON', 'SEASONderiv', 'SEASONpart', 'FOLD', 'FOLDderiv', 'FOLDpart', 'OCEAN', 'OCEANderiv', 'OCEANpart', 'WEIGHT', 'WEIGHTderiv', 'WEIGHTpart', 'TIME', 'TIMEderiv', 'TIMEpart', 'LOCATIONOTHER', 'LOCATIONOTHERderiv', 'LOCATIONOTHERpart', 'DISEASE', 'DISEASEderiv', 'DISEASEpart', 'EVENT', 'EVENTderiv', 'EVENTpart', 'CITYSTATE', 'CITYSTATEderiv', 'CITYSTATEpart', 'WOA', 'WOAderiv', 'WOApart', 'SPORTSTEAM', 'SPORTSTEAMderiv', 'SPORTSTEAMpart', 'DATEOTHER', 'DATEOTHERderiv', 'DATEOTHERpart', 'GRPPER', 'GRPPERderiv', 'GRPPERpart', 'NAMEMOD', 'NAMEMODderiv', 'NAMEMODpart', 'BOOK', 'BOOKderiv', 'BOOKpart', 'ELECTRONICS', 'ELECTRONICSderiv', 'ELECTRONICSpart', 'ARMY', 'ARMYderiv', 'ARMYpart', 'FACILITY', 'FACILITYderiv', 'FACILITYpart', 'PRODUCTDRUG', 'PRODUCTDRUGderiv', 'PRODUCTDRUGpart', 'TVSHOW', 'TVSHOWderiv', 'TVSHOWpart', 'HURRICANE', 'HURRICANEderiv', 'HURRICANEpart', 'SPORTSEVENT', 'SPORTSEVENTderiv', 'SPORTSEVENTpart', 'NICKNAME', 'NICKNAMEderiv', 'NICKNAMEpart', 'FILM', 'FILMderiv', 'FILMpart', 'LANGUAGE', 'LANGUAGEderiv', 'LANGUAGEpart', 'PRODUCTFOOD', 'PRODUCTFOODderiv', 'PRODUCTFOODpart', 'RELIGION', 'RELIGIONderiv', 'RELIGIONpart', 'SUBURB', 'SUBURBderiv', 'SUBURBpart', 'GRPLOC', 'GRPLOCderiv', 'GRPLOCpart', 'SONG', 'SONGderiv', 'SONGpart', 'QUANTITYOTHER', 'QUANTITYOTHERderiv', 'QUANTITYOTHERpart', 'SPACE', 'SPACEderiv', 'SPACEpart', 'WAR', 'WARderiv', 'WARpart', 'RIVER', 'RIVERderiv', 'RIVERpart', 'CHEMICAL', 'CHEMICALderiv', 'CHEMICALpart', 'FUND', 'FUNDderiv', 'FUNDpart', 'BRIDGE', 'BRIDGEderiv', 'BRIDGEpart', 'HOTEL', 'HOTELderiv', 'HOTELpart', 'PLAY', 'PLAYderiv', 'PLAYpart', 'STADIUM', 'STADIUMderiv', 'STADIUMpart', 'AWARD', 'AWARDderiv', 'AWARDpart', 'ORGRELIGIOUS', 'ORGRELIGIOUSderiv', 'ORGRELIGIOUSpart', 'AIRPORT', 'AIRPORTderiv', 'AIRPORTpart', 'GOD', 'GODderiv', 'GODpart', 'ANIMATE', 'ANIMATEderiv', 'ANIMATEpart', 'ATTRACTION', 'ATTRACTIONderiv', 'ATTRACTIONpart', 'HOSPITAL', 'HOSPITALderiv', 'HOSPITALpart', 'WEAPON', 'WEAPONderiv', 'WEAPONpart', 'MUSEUM', 'MUSEUMderiv', 'MUSEUMpart', 'ENERGY', 'ENERGYderiv', 'ENERGYpart', 'PAINTING', 'PAINTINGderiv', 'PAINTINGpart', 'SPEED', 'SPEEDderiv', 'SPEEDpart', 'BAND', 'BANDderiv', 'BANDpart', 'SPORTSSEASON', 'SPORTSSEASONderiv', 'SPORTSSEASONpart', 'SCINAME', 'SCINAMEderiv', 'SCINAMEpart', 'ADDRESSNON', 'ADDRESSNONderiv', 'ADDRESSNONpart', 'ALBUM', 'ALBUMderiv', 'ALBUMpart', 'CONCERT', 'CONCERTderiv', 'CONCERTpart', 'NATURALDISASTER', 'NATURALDISASTERderiv', 'NATURALDISASTERpart', 'TEMPERATURE', 'TEMPERATUREderiv', 'TEMPERATUREpart', 'BORDER', 'BORDERderiv', 'BORDERpart', 'CHANNEL', 'CHANNELderiv', 'CHANNELpart', 'STATION', 'STATIONderiv', 'STATIONpart'}


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
