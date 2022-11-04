from allennlp.data.fields.text_field import TextField
from dataset_reader import tne_reader
from allennlp.data import DatasetReader, Instance, Vocabulary, TextFieldTensors, DataLoader
from modeling.models import tne_coupled
from typing import Dict, Iterable, List
from allennlp.data.token_indexers.pretrained_transformer_mismatched_indexer import PretrainedTransformerMismatchedIndexer
from allennlp.models import Model
from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import PretrainedTransformerMismatchedEmbedder
from allennlp.modules.seq2seq_encoders.pass_through_encoder import PassThroughEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation
from allennlp.nn.initializers import *
from torch.nn.init import *
from torch.nn.modules.activation import *
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler
from allennlp.data.data_loaders import *
from allennlp.training.trainer import Trainer
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer, AdamOptimizer
from allennlp.training.learning_rate_schedulers.slanted_triangular import SlantedTriangular
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                          'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                          'inside', 'outside', 'into', 'around']

num_lables = len(preposition_list)


train_data_path =  'data/train.jsonl'
validation_data_path = 'data/dev.jsonl'
test_data_path = 'data/test_unlabeled.jsonl'


transformer_dim = 1024;  # uniquely determined by transformer_model
max_length = 512
span_embedding_dim = 2 * transformer_dim

pretrained_indexer = PretrainedTransformerMismatchedIndexer(model_name="SpanBERT/spanbert-large-cased", max_length=max_length)

indexer_dict = {"tokens": pretrained_indexer}

d_reader = tne_reader.TNEReader(prepositions=preposition_list, development=False, token_indexers=indexer_dict)

# d_reader = tne_reader.TNEReader(prepositions=preposition_list, development=False)

validation_reader = tne_reader.TNEReader(prepositions=preposition_list, token_indexers=indexer_dict)

# validation_reader = tne_reader.TNEReader(prepositions=preposition_list)

# print(d_reader._token_indexers)

train_instances = list(d_reader._read(train_data_path))

validation_instances = list(validation_reader._read(validation_data_path))


# for i in train_instances[:1]:
#   print(i)

#get allennlp vocab

vocab = Vocabulary.from_instances(train_instances)

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")

    regexes =  [
        (".*_span_updating_gated_sum.*weight", XavierNormalInitializer()),
        (".*linear_layers.*weight", XavierNormalInitializer()),
        (".*scorer.*weight", XavierNormalInitializer()),
        ("_distance_embedding.weight", XavierNormalInitializer()),
        ("_span_width_embedding.weight", XavierNormalInitializer()),
        ("_context_layer._module.weight_ih.*",XavierNormalInitializer()),
        ("_context_layer._module.weight_hh.*", OrthogonalInitializer())
      ]
    
    vocab_size = vocab.get_vocab_size("tokens")
    
    pretrained_embedder = PretrainedTransformerMismatchedEmbedder(model_name="SpanBERT/spanbert-large-cased", max_length=max_length)

    context_layer = PassThroughEncoder(input_dim = transformer_dim)

    anchor_feedforward = FeedForward(input_dim=span_embedding_dim, num_layers=2, hidden_dims=500, activations=Activation.by_name('relu')(),dropout=0.3)

    complement_feedforward = FeedForward(input_dim=span_embedding_dim, num_layers=2, hidden_dims=500, activations=Activation.by_name('relu')(),dropout=0.3)

    preposition_predictor = FeedForward(input_dim=500+500, num_layers=2, hidden_dims=[100, num_lables], activations=
                                        [Activation.by_name('relu')(), Activation.by_name('linear')()],dropout=0.3)
    
    freeze = False

    initliazer = InitializerApplicator(regexes=regexes)

    return tne_coupled.TNECoupledModel(vocab, pretrained_embedder, context_layer, anchor_feedforward, complement_feedforward, preposition_predictor, preposition_list, initializer=initliazer, freeze=freeze)
    # return tne_coupled.TNECoupledModel(vocab, pretrained_embedder, context_layer, anchor_feedforward, complement_feedforward, preposition_predictor, preposition_list, freeze=freeze)

model = build_model(vocab)

# outputs = model.forward_on_instances(train_instances[2:4])
# print(output)

#need to verify MultiProcessDataLoader
def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    bucketSampler = BucketBatchSampler(batch_size=1, sorting_keys=["text"])
    train_loader = MultiProcessDataLoader(d_reader, train_data_path, batch_sampler=bucketSampler)
    dev_loader = MultiProcessDataLoader(validation_reader, validation_data_path, batch_sampler=bucketSampler)
    return train_loader, dev_loader

train_loader, dev_loader = build_data_loaders(train_instances, validation_instances)

train_loader.index_with(vocab)
dev_loader.index_with(vocab)


# def build_trainer(
#     model: Model,
#     serialization_dir: str,
#     train_loader: DataLoader,
#     dev_loader: DataLoader,
# ) -> Trainer:
#     parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
#     parameter_groups = [([".*transformer.*"], {"lr": 1e-5})]
#     optimizer = HuggingfaceAdamWOptimizer(parameters, parameter_groups, lr=1e-5)
#     num_epochs = 40
#     patience = 10
#     cuda_device = -1
#     validation_metric = "+overall_f1"

#     learning_rate_scheduler = SlantedTriangular(optimizer, num_epochs, cut_frac=0.06)

#     trainer = GradientDescentTrainer(
#         model=model,
#         optimizer = optimizer,
#         data_loader = train_loader,
#         validation_data_loader = dev_loader,
#         patience = patience,
#         validation_metric=validation_metric,
#         num_epochs=num_epochs,
#         cuda_device = cuda_device,
#         learning_rate_scheduler = learning_rate_scheduler
#         )
    
#     return trainer

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer

trainer = build_trainer(model, "./", train_loader, dev_loader)
print("Starting training")
trainer.train()
print("Finished training")