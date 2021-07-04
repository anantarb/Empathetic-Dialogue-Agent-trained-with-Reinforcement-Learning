"""
Use pretrained empathy classifiers to predict empathy scores (emotional reactions, interpretations, explorations)
"""
from empathy_mental_health.src.models.models import BiEncoderAttentionWithRationaleClassification

import os
import torch
import numpy as np
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler


trained_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models')


class EmpathyClassifier:
    def __init__(self, run_on_cpu=False):
        self.trained_models_path = trained_models_dir
        self.device = self.torch_device_setup() if not run_on_cpu else torch.device('cpu')
        self.model_suite = self.load_model_suite()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

    def torch_device_setup(self):
        """Use GPU if available"""
        if torch.cuda.is_available():
            print('Running on GPU.')
            device = torch.device("cuda")
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        return device

    def load_model_suite(self):
        """Load models"""
        print('Loading trained models:')
        dropout_default = 0.1

        print('Loading emotional reactions model ...')
        emotional_reactions_model = BiEncoderAttentionWithRationaleClassification(hidden_dropout_prob=dropout_default)
        emotional_reactions_model.load_state_dict(
            torch.load(os.path.join(self.trained_models_path, 'emotional_reactions.pth'),
                       map_location=self.device))
        emotional_reactions_model = emotional_reactions_model.to(self.device)
        emotional_reactions_model.eval()
        print('Successfully loaded trained emotional reactions model.')

        print('Loading interpretations model ...')
        interpretations_model = BiEncoderAttentionWithRationaleClassification(hidden_dropout_prob=dropout_default)
        interpretations_model.load_state_dict(torch.load(os.path.join(self.trained_models_path, 'interpretations.pth'),
                                                         map_location=self.device))
        interpretations_model = interpretations_model.to(self.device)
        interpretations_model.eval()
        print('Successfully loaded trained interpretations model.')

        print('Loading explorations model ...')
        explorations_model = BiEncoderAttentionWithRationaleClassification(hidden_dropout_prob=dropout_default)
        explorations_model.load_state_dict(torch.load(os.path.join(self.trained_models_path, 'explorations.pth'),
                                                      map_location=self.device))
        explorations_model = explorations_model.to(self.device)
        explorations_model.eval()
        print('Successfully loaded trained explorations model.')

        trained_empathy_model_suite = [emotional_reactions_model, interpretations_model, explorations_model]

        '''
        Do not finetune seeker encoder
        '''
        for model in trained_empathy_model_suite:
            params = list(model.named_parameters())
            for p in model.seeker_encoder.parameters():
                p.requires_grad = False

        return [emotional_reactions_model, interpretations_model, explorations_model]

    def compute_empathy_levels(self, utterance_pairs):
        """
        Computes the levels of the three empathy mechanisms (emotional reactions, interpretations, explorations)
        for a batch of utterance pairs.
        Input (utterance_pairs): 2d ndarray of utterance pairs (first/second column contains first/second utterance)
        """

        # first utterances (seeker)
        tokenizer_first_utts = self.tokenizer.batch_encode_plus(utterance_pairs[:, 0],
                                                                add_special_tokens=True, max_length=64,
                                                                pad_to_max_length=True, return_attention_mask=True)
        input_ids_first_utts = torch.tensor(tokenizer_first_utts['input_ids'])
        attention_masks_first_utts = torch.tensor(tokenizer_first_utts['attention_mask'])

        # second utterances (response)
        tokenizer_second_utts = self.tokenizer.batch_encode_plus(utterance_pairs[:, 1],
                                                                 add_special_tokens=True, max_length=64,
                                                                 pad_to_max_length=True, return_attention_mask=True)
        input_ids_second_utts = torch.tensor(tokenizer_second_utts['input_ids'])
        attention_masks_second_utts = torch.tensor(tokenizer_second_utts['attention_mask'])

        # dummy_labels = torch.zeros([len(utterance_pairs)], dtype=torch.int64)
        # dummy_rationales = torch.zeros([len(utterance_pairs), 64], dtype=torch.int64)
        # dummy_rationales_trimmed = torch.zeros([len(utterance_pairs)], dtype=torch.int64)
        #
        # tensor_dataset = TensorDataset(input_ids_first_utts, attention_masks_first_utts, input_ids_second_utts,
        #                              attention_masks_second_utts, dummy_labels, dummy_rationales, dummy_rationales_trimmed)
        # dataloader = DataLoader(tensor_dataset, sampler=SequentialSampler(tensor_dataset), batch_size=len(utterance_pairs))
        #
        # batch = next(iter(dataloader))
        #
        # b_input_ids_SP = batch[0].to(self.device)
        # b_input_mask_SP = batch[1].to(self.device)
        # b_input_ids_RP = batch[2].to(self.device)
        # b_input_mask_RP = batch[3].to(self.device)
        # b_labels = batch[4].to(self.device)
        # b_rationales = batch[5].to(self.device)

        predicted_empathy_levels = []

        with torch.no_grad():
            for model in self.model_suite:
                _, _, _, logits_empathy, _ = model(
                    input_ids_SP=input_ids_first_utts,#b_input_ids_SP,
                    input_ids_RP=input_ids_second_utts,#b_input_ids_RP,
                    attention_mask_SP=attention_masks_first_utts,#b_input_mask_SP,
                    attention_mask_RP=attention_masks_second_utts)#b_input_mask_RP,)

                # get predicted empathy labels
                logits_empathy = logits_empathy.detach().cpu().numpy()
                predicted_empathy_levels.append(np.argmax(logits_empathy, axis=1))

        return np.array(predicted_empathy_levels).T