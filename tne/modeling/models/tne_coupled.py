import logging
from typing import Any, Dict, List

import numpy as np
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics.f1_measure import F1Measure
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from overrides import overrides

from tne.modeling.metrics.mcf1_measure import MCF1Measure

logger = logging.getLogger(__name__)


@Model.register("tne_coupled_model")
class TNECoupledModel(Model):
    """
    Details

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the `text` `TextField` we get as input to the model.
    context_layer : `Seq2SeqEncoder`
        This layer incorporates contextual information for each word in the document.
    anchor_feedforward : `FeedForward`
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    complement_feedforward : `FeedForward`
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    max_span_width : `int`
        The maximum width of candidate spans.
    coarse_to_fine: `bool`, optional (default = `False`)
        Whether or not to apply the coarse-to-fine filtering.
    inference_order: `int`, optional (default = `1`)
        The number of inference orders. When greater than 1, the span representations are
        updated and coreference scores re-computed.
    lexical_dropout : `int`
        The probability of dropping out dimensions of the embedded text.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            context_layer: Seq2SeqEncoder,
            anchor_feedforward: FeedForward,
            complement_feedforward: FeedForward,
            preposition_predictor: FeedForward,
            prepositions: List[str],
            lexical_dropout: float = 0.2,
            initializer: InitializerApplicator = InitializerApplicator(),
            freeze: bool = False,
            **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._prepositions = prepositions
        self._freeze = freeze

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._anchor_feedforward = TimeDistributed(anchor_feedforward)
        self._complement_feedforward = TimeDistributed(complement_feedforward)

        self._preposition_scorer = TimeDistributed(preposition_predictor)

        self._endpoint_span_extractor = EndpointSpanExtractor(
            context_layer.get_output_dim(),
            combination="x,y",
            num_width_embeddings=None,
            span_width_embedding_dim=None,
            bucket_widths=False,
        )

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        initializer(self)

        self.preposition_loss = torch.nn.CrossEntropyLoss()

        self.link_f1 = F1Measure(positive_label=1)
        self.micro_f1 = FBetaMeasure(beta=1.0, average='micro', labels=list(range(len(prepositions))))
        self.overall_f1 = MCF1Measure()
        self.prep_acc = CategoricalAccuracy()
        self.identified_gold_prep_acc = CategoricalAccuracy()
        self.non_identified_gold_prep_acc = CategoricalAccuracy()

    @overrides
    def forward(
            self,  # type: ignore
            text: TextFieldTensors,
            spans: torch.IntTensor,
            metadata: List[Dict[str, Any]] = None,
            link_labels: torch.IntTensor = None,
            preposition_labels: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        text : `TextFieldTensors`, required.
            The output of a `TextField` representing the text of
            the document.
        spans : `torch.IntTensor`, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a `ListField[SpanField]` of
            indices into the text of the document.
        span_labels : `torch.IntTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans), representing the cluster ids
            of each span, or -1 for those which do not appear in any clusters.
        metadata : `List[Dict[str, Any]]`, optional (default = `None`).
            A metadata dictionary for each instance in the batch. We use the "original_text" and "clusters" keys
            from this dictionary, which respectively have the original text and the annotated gold coreference
            clusters for that instance.

        # Returns

        An output dictionary consisting of:

        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        if self._freeze:
            with torch.no_grad():
                span_embeddings = self.get_span_embeddings(text, spans)
        else:
            span_embeddings = self.get_span_embeddings(text, spans)

        anchor_reps = self._anchor_feedforward(span_embeddings)
        complement_reps = self._complement_feedforward(span_embeddings)

        # Creating a large matrix that concatenates all permutations of spans with one another, between the
        #  representation obtained from the anchor representations and the antecedent representations
        # The reshape corresponds to the number of entities to the power of two,
        #  and the the representation size of the antecedent + the anchor
        mat = torch.cat([anchor_reps.squeeze(0).unsqueeze(1).repeat(1, anchor_reps.shape[1], 1),
                         complement_reps.squeeze(0).repeat(anchor_reps.shape[1], 1, 1)], -1) \
            .reshape(anchor_reps.shape[1] ** 2, anchor_reps.shape[-1] * 2)

        preposition_scores = self._preposition_scorer(mat.unsqueeze(0)).squeeze(0)

        preposition_hat = torch.argmax(preposition_scores, dim=1).unsqueeze(0)

        output_dict = {
            "predicted_prepositions": preposition_hat,
        }
        if preposition_labels is not None:
            preposition_labels = preposition_labels.reshape(-1)
            preposition_loss = self.preposition_loss(preposition_scores, preposition_labels)

            output_dict["loss"] = preposition_loss

            extended_prepositions = [x['extended_prepositions'] for x in metadata][0]
            adapted_preposition_labels = self.adapt_prep_labels_to_predictions(preposition_labels,
                                                                               extended_prepositions, preposition_hat)

            lowest_val = preposition_scores.detach().cpu().min().numpy().tolist() - 1
            lowest_vec = torch.ones_like(preposition_labels) * lowest_val
            preposition_adapted_scores = preposition_scores.clone().detach()
            preposition_adapted_scores[:, 0] = lowest_vec
            output_dict['best_prepositions'] = torch.argmax(preposition_adapted_scores, dim=1).unsqueeze(0)

            self.prep_acc(preposition_adapted_scores.unsqueeze(0),
                          adapted_preposition_labels.unsqueeze(0),
                          (link_labels != -1) &
                          (adapted_preposition_labels != 0).unsqueeze(0))

            self.identified_gold_prep_acc(preposition_scores.unsqueeze(0),
                                          adapted_preposition_labels.unsqueeze(0),
                                          ((link_labels != -1) &
                                           (adapted_preposition_labels != 0).unsqueeze(0) & (preposition_hat != 0)))

            self.non_identified_gold_prep_acc(preposition_adapted_scores.unsqueeze(0),
                                              adapted_preposition_labels.unsqueeze(0),
                                              (link_labels != -1) &
                                              ((adapted_preposition_labels != 0).unsqueeze(0) & (preposition_hat == 0)))

            self.measure_overal_f1(preposition_scores, adapted_preposition_labels, link_labels)

            is_relation_preds = torch.cat([(preposition_hat == 0).int(), (preposition_hat != 0).int()], dim=0).T
            self.link_f1(is_relation_preds, link_labels.squeeze(0),
                         (link_labels != -1).squeeze(0))

            self.overall_f1(preposition_hat,
                            adapted_preposition_labels,
                            (link_labels != -1))

        if metadata is not None:
            output_dict["document"] = [x["original_text"] for x in metadata]
            output_dict["tokenized_entities"] = [x["tokenized_entities"] for x in metadata]
            output_dict["tokens"] = [x["tokens"] for x in metadata]
        return output_dict

    def get_span_embeddings(self, text, spans):
        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text)

        # Shape: (batch_size, document_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim)
        span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)

        return span_embeddings

    def measure_overal_f1(self, prep_scores, prep_labels, link_labels):
        # tuples that the model predicted there is a link
        link_exist = prep_scores.detach().argmax(dim=1) != 0

        self.micro_f1(prep_scores.unsqueeze(0),
                      prep_labels.unsqueeze(0),
                      # in cases where both the model predicted no-link, and gold label indicate no-link, masking,
                      # to not count these points
                      # also masking self-link
                      (((link_labels == 1) | link_exist) & (link_labels != -1)))

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1 = self.link_f1.get_metric(reset)
        prep_acc = self.prep_acc.get_metric(reset)
        identified_prep_acc = self.identified_gold_prep_acc.get_metric(reset)
        non_identified_prep_acc = self.non_identified_gold_prep_acc.get_metric(reset)
        micro_f1 = self.micro_f1.get_metric(reset)
        overall_f1 = self.overall_f1.get_metric(reset)

        return {
            "links_p": f1['precision'],
            "links_r": f1['recall'],
            "links_f1": f1['f1'],
            "preposition_acc": prep_acc,
            "identified_prep_acc": identified_prep_acc,
            "non_identified_prep_acc": non_identified_prep_acc,
            'overall_micro_f1': micro_f1['fscore'],
            'overall_p': overall_f1['precision'],
            'overall_r': overall_f1['recall'],
            'overall_f1': overall_f1['fscore'],
            # The precision and recall are the same as the f1 due to the micro averaging, so not logging them
        }

    @staticmethod
    def adapt_prep_labels_to_predictions(preposition_labels, extended_prepositions_labels, preposition_predictions):
        updated_labels = []
        for prep_l, prep_p, prep_extend in zip(preposition_labels, preposition_predictions.squeeze(0),
                                               extended_prepositions_labels):
            if prep_l == prep_p:
                updated_labels.append(prep_p)
            elif prep_p in prep_extend:
                updated_labels.append(prep_p)
            else:
                updated_labels.append(prep_l)
        return torch.tensor(updated_labels).to(preposition_labels.device)

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Takes the result of `forward` and makes it human readable.

        This method `modifies` the input dictionary, and returns a human interpretable version, where all the non-links
        predictions are removed, and added with the relevant preposition

        """
        # links = output_dict['predicted_links'][0]
        preps = output_dict['predicted_prepositions'][0]
        entities = output_dict['tokenized_entities'][0]
        nps_list = list(entities.keys())
        n = int(np.sqrt(len(preps)))
        predictions = {}
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if int(preps[i * n + j]) != 0:
                    predictions[(nps_list[i], nps_list[j])] = self._prepositions[int(preps[i * n + j])]

        tokens = output_dict['tokens'][0]

        human_links = []
        full_links = []
        for (from_entity, to_entity), prep in predictions.items():
            from_start = entities[from_entity]['first_token']
            from_end = entities[from_entity]['last_token']
            from_str = tokens[from_start: from_end + 1]

            to_start = entities[to_entity]['first_token']
            to_end = entities[to_entity]['last_token']
            to_str = tokens[to_start: to_end + 1]

            human_links.append(' '.join(from_str + ['*' + prep + '*'] + to_str))

            full_links.append({
                'from_first': from_start,
                'from_last': from_end,
                'to_first': to_start,
                'to_last': to_end,
                'preposition': prep,
            })

        entities_str = []
        for ind, ent in entities.items():
            start = ent['first_token']
            end = ent['last_token']
            ent_str = ' '.join(tokens[start: end + 1])
            entities_str.append(ent_str)

        human_output = {'document': output_dict['document'],
                        'entities': [entities_str],
                        'predicted_links': [human_links],
                        'full_links': [full_links],
                        'predicted_prepositions': [preps],
                        }

        return human_output
