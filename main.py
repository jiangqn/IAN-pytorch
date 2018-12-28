import argparse
from model import *
from utils import *
import os
import time
from torch.utils.data import DataLoader
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', default=300)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--n_epoch', default=30)
parser.add_argument('--hidden_size', default=300)
parser.add_argument('--n_class', default=3)
parser.add_argument('--pre_processed', default=True)
parser.add_argument('--learning_rate', default=0.01)
parser.add_argument('--l2_reg', default=1e-5)
parser.add_argument('--dropout', default=0.01)
parser.add_argument('--max_aspect_len', default=0)
parser.add_argument('--max_context_len', default=0)
parser.add_argument('--dataset', default='data/laptop/')
parser.add_argument('--embedding_file_name', default='data/glove.840B.300d.txt')
parser.add_argument('--embedding', default=0)
parser.add_argument('--vocab_size', default=0)

config = parser.parse_args()

def main():
    start_time = time.time()
    print('Loading data info ...')
    word2id, config.max_aspect_len, config.max_context_len = get_data_info(config.dataset, config.pre_processed)
    config.vocab_size = len(word2id)
    train_data = read_data(word2id, config.max_aspect_len, config.max_context_len, config.dataset + 'train',
                           config.pre_processed)
    test_data = read_data(word2id, config.max_aspect_len, config.max_context_len, config.dataset + 'test',
                          config.pre_processed)
    print('Loading pre-trained word vectors ...')
    config.embedding = load_word_embeddings(config.embedding_file_name, config.embedding_size, word2id)
    train_dataset = IanDataset(train_data)
    test_dataset = IanDataset(test_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    model = IAN(config).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    max_acc = 0
    for epoch in range(config.n_epoch):
        train_total_cases = 0
        train_correct_cases = 0
        for data in train_loader:
            aspects, contexts, labels, aspect_masks, context_masks = data
            aspects, contexts, labels = aspects.cuda(), contexts.cuda(), labels.cuda()
            aspect_masks, context_masks = aspect_masks.cuda(), context_masks.cuda()
            optimizer.zero_grad()
            outputs = model(aspects, contexts, aspect_masks, context_masks)
            _, predicts = outputs.max(dim=1)
            train_total_cases += labels.shape[0]
            train_correct_cases += (predicts == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_accuracy = train_correct_cases / train_total_cases
        test_total_cases = 0
        test_correct_cases = 0
        for data in test_loader:
            aspects, contexts, labels, aspect_masks, context_masks = data
            aspects, contexts, labels = aspects.cuda(), contexts.cuda(), labels.cuda()
            aspect_masks, context_masks = aspect_masks.cuda(), context_masks.cuda()
            outputs = model(aspects, contexts, aspect_masks, context_masks)
            _, predicts = outputs.max(dim=1)
            test_total_cases += labels.shape[0]
            test_correct_cases += (predicts == labels).sum().item()
        test_accuracy = test_correct_cases / test_total_cases
        print('[epoch %03d] train accuracy: %.4f test accuracy: %.4f' % (epoch, train_accuracy, test_accuracy))
        max_acc = max(max_acc, test_accuracy)
    print('max test accuracy:', max_acc)
    end_time = time.time()
    print('Time Costing: %s' % (end_time - start_time))

if __name__ == '__main__':
    main()