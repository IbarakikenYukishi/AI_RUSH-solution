import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
from model import Baseline, Resnet18, Resnet50, Alex, Inception, Vgg
import nsml
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import train_dataloader
from dataloader import AIRushDataset


def to_np(t):
    return t.cpu().detach().numpy()


def bind_model(model_list, index, MODELNUM):
    def save(dir_name, **kwargs):
        # 保存パスの指定
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')

        print('savepath:' + save_state_path)

        # 保存するモデルについての設定
        state = {
            'model0': model_list[0].state_dict(),
            'model1': model_list[1].state_dict(),
            'model2': model_list[2].state_dict(),
            'model3': model_list[3].state_dict(),
            # ここになんか書き足せばいける?
        }
        torch.save(state, save_state_path)

    def load(dir_name):
        load_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = torch.load(load_state_path)
        print('loadpath:' + load_state_path)
        print('index is ' + str(index))
        if index == -1:
            model_list[0].load_state_dict(state['model0'])
            model_list[1].load_state_dict(state['model1'])
            model_list[2].load_state_dict(state['model2'])
            model_list[3].load_state_dict(state['model3'])            
        else:
            model_list[index].load_state_dict(state['model'])

    def infer(test_image_data_path, test_meta_data_path):
        print('ensemble dekitennzyane?')
        # DONOTCHANGE This Line
        test_meta_data = pd.read_csv(
            test_meta_data_path, delimiter=',', header=0)

        input_size = 128  # you can change this according to your model.
        input_size_resnet = 224
        input_size_inception = 299
        # you can change this. But when you use 'nsml submit --test' for test
        # infer, there are only 200 number of data.
        batch_size =62
        device = 0

        # ここのローディングもいくつかバリエーション持たせてtransformの変換でうまいことできるかも
        dataloader_resnet = DataLoader(
            AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                          transform=transforms.Compose([transforms.Resize((input_size_resnet, input_size_resnet)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

        dataloader_inception = DataLoader(
            AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                          transform=transforms.Compose([transforms.Resize((input_size_inception, input_size_inception)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

        dataloader_resnet_crop = DataLoader(
            AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                      transform=transforms.Compose([transforms.Resize((input_size_resnet+20, input_size_resnet+20)),
                                                    transforms.RandomCrop(
                                                        input_size_resnet),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)


        for __modelnum__ in range(MODELNUM):
            model_list[__modelnum__].to(device)
            model_list[__modelnum__].eval()

        predict_list = []
        for batch_idx, (image_resnet, image_inception, image_resnet_crop) in enumerate(zip(dataloader_resnet, dataloader_inception,dataloader_resnet_crop)):
            image_resnet = image_resnet.to(device)
            image_inception = image_inception.to(device)
            image_resnet_crop = image_resnet_crop.to(device)

            output0 = to_np(model_list[0](image_resnet).double())  # VGG

            output1 = to_np(model_list[1](
                image_inception).squeeze(-1).squeeze(-1).double())  # Inception

            output2 = to_np(model_list[2](image_resnet).double())  # Resnet

            output3 = to_np(model_list[3](image_resnet_crop).double())  # Resnet

            print(output0.shape)
            print(output1.shape)
            print(output2.shape)
            print(output3.shape)            

            print(np.sum(output0, axis=1))
            print(np.sum(output1, axis=1))
            print(np.sum(output2, axis=1))
            print(np.sum(output3, axis=1))

            output0_sum = np.abs(np.sum(output0, axis=1))
            output1_sum = np.abs(np.sum(output1, axis=1))
            output2_sum = np.abs(np.sum(output2, axis=1))
            output3_sum = np.abs(np.sum(output3, axis=1))

            print(output0_sum.shape)
            print(output1_sum.shape)
            print(output2_sum.shape)
            print(output3_sum.shape)

            output0 /= output0_sum[:, np.newaxis]
            output1 /= output1_sum[:, np.newaxis]
            output2 /= output2_sum[:, np.newaxis]
            output3 /= output3_sum[:, np.newaxis]

            print(np.sum(output0[0, :]))
            print(np.sum(output1[0, :]))
            print(np.sum(output2[0, :]))
            print(np.sum(output3[0, :]))

            output2*=0.666 #出力が負なので、低い方が重みついてる

            output = output0 + output1 + output2 + output3

            predict = np.argmax(output, axis=1)

            predict_list.append(predict)

        predict_vector = np.concatenate(predict_list, axis=0)
        # this return type should be a numpy array which has shape of (138343)
        return predict_vector

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=1)

    # custom args
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--resnet', default=False)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--output_size', type=int, default=350)  # Fixed
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    model0 = Vgg(args.output_size)
    model1 = Inception(args.output_size)
    model2 = Resnet50(args.output_size)
    model3 = Resnet18(args.output_size)

    # モデルの数に応じて適宜変える
    model_list = [model0, model1, model2, model3]

    optimizer0 = optim.Adam(model_list[0].parameters(), args.learning_rate)
    criterion0 = nn.CrossEntropyLoss()  # multi-class classification task
    model_list[0] = model_list[0].to(device)
    model_list[0].train()

    optimizer1 = optim.Adam(model_list[1].parameters(), args.learning_rate)
    criterion1 = nn.CrossEntropyLoss()  # multi-class classification task
    model_list[1] = model_list[1].to(device)
    model_list[1].train()

    optimizer1 = optim.Adam(model_list[2].parameters(), args.learning_rate)
    criterion1 = nn.CrossEntropyLoss()  # multi-class classification task
    model_list[2] = model_list[2].to(device)
    model_list[2].train()

    optimizer1 = optim.Adam(model_list[3].parameters(), args.learning_rate)
    criterion1 = nn.CrossEntropyLoss()  # multi-class classification task
    model_list[3] = model_list[3].to(device)
    model_list[3].train()

    MODELNUM = 4

    bind_model(model_list, 0, MODELNUM)
    nsml.load(checkpoint='7', session='team_97/airush1/261')  # Vgg
    bind_model(model_list, 1, MODELNUM)
    nsml.load(checkpoint='9', session='team_97/airush1/236')  # Inception
    bind_model(model_list, 2, MODELNUM)
    nsml.load(checkpoint='25', session='team_97/airush1/94')  # resnet50
    bind_model(model_list, 3, MODELNUM)
    nsml.load(checkpoint='20', session='team_97/airush1/164')  # resnet18

    bind_model(model_list, -1, MODELNUM)
    print('nsml.save suruyo')
    nsml.save('ensemble')

    bTrainmode = False

    if args.pause:
        print('Finished')
        nsml.paused(scope=locals())
    if args.mode == "train":
        bTrainmode = True

        # Warning: Do not load data before this line
        dataloader = train_dataloader(
            args.input_size, args.batch_size, args.num_workers)

        # 学習
        for epoch_idx in range(1, args.epochs + 1):
            total_loss0 = 0
            total_correct0 = 0
            total_loss1 = 0
            total_correct1 = 0
            total_loss2 = 0
            total_correct2 = 0
            total_loss3 = 0
            total_correct3 = 0

            for batch_idx, (image, tags) in enumerate(dataloader):
                # 一連のモデル学習
                image = image.to(device)
                tags = tags.to(device)

                # 各モデルの訓練
                optimizer0.zero_grad()
                output0 = model_list[0](image).double()
                loss0 = criterion0(output0, tags)
                loss0.backward()
                optimizer0.step()

                optimizer1.zero_grad()
                output1 = model_list[1](image).double()
                loss1 = criterion1(output1, tags)
                loss1.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                output2 = model_list[2](image).double()
                loss2 = criterion2(output2, tags)
                loss2.backward()
                optimizer2.step()

                optimizer3.zero_grad()
                output3 = model_list[3](image).double()
                loss3 = criterion3(output3, tags)
                loss3.backward()
                optimizer3.step()

                label_vector = to_np(tags)
                # 予測
                output_prob0 = F.softmax(output0, dim=1)
                predict_vector0 = np.argmax(to_np(output_prob0), axis=1)
                bool_vector0 = predict_vector0 == label_vector
                accuracy0 = bool_vector0.sum() / len(bool_vector0)

                output_prob1 = F.softmax(output1, dim=1)
                predict_vector1 = np.argmax(to_np(output_prob1), axis=1)
                bool_vector1 = predict_vector1 == label_vector
                accuracy1 = bool_vector1.sum() / len(bool_vector1)

                output_prob2 = F.softmax(output2, dim=1)
                predict_vector2 = np.argmax(to_np(output_prob2), axis=1)
                bool_vector2 = predict_vector2 == label_vector
                accuracy2 = bool_vector2.sum() / len(bool_vector2)

                output_prob3 = F.softmax(output3, dim=1)
                predict_vector3 = np.argmax(to_np(output_prob3), axis=1)
                bool_vector3 = predict_vector3 == label_vector
                accuracy3 = bool_vector3.sum() / len(bool_vector3)

                if batch_idx % args.log_interval == 0:
                    print('Model0:Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f}'.format(batch_idx,
                                                                                                len(
                                                                                                    dataloader),
                                                                                                loss0.item(),
                                                                                                accuracy0))
                    print('Model1:Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f}'.format(batch_idx,
                                                                                                len(
                                                                                                    dataloader),
                                                                                                loss1.item(),
                                                                                                accuracy1))
                    print('Model2:Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f}'.format(batch_idx,
                                                                                                len(
                                                                                                    dataloader),
                                                                                                loss2.item(),
                                                                                                accuracy2))
                    print('Model3:Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f}'.format(batch_idx,
                                                                                                len(
                                                                                                    dataloader),
                                                                                                loss3.item(),
                                                                                                accuracy3))
                total_loss0 += loss0.item()
                total_correct0 += bool_vector0.sum()
                total_loss1 += loss1.item()
                total_correct1 += bool_vector1.sum()
                total_loss2 += loss2.item()
                total_correct2 += bool_vector2.sum()
                total_loss3 += loss3.item()
                total_correct3 += bool_vector3.sum()

            nsml.save(epoch_idx)

            # 結果報告
            print('Model0:Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f}'.format(epoch_idx,
                                                                                  args.epochs,
                                                                                  total_loss0 /
                                                                                  len(
                                                                                      dataloader.dataset),
                                                                                  total_correct0 / len(dataloader.dataset)))
            nsml.report(
                summary=True,
                step=epoch_idx,
                scope=locals(),
                **{
                    "train__Loss": total_loss0 / len(dataloader.dataset),
                    "train__Accuracy": total_correct0 / len(dataloader.dataset),
                })

            print('Model1:Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f}'.format(epoch_idx,
                                                                                  args.epochs,
                                                                                  total_loss1 /
                                                                                  len(
                                                                                      dataloader.dataset),
                                                                                  total_correct1 / len(dataloader.dataset)))
            nsml.report(
                summary=True,
                step=epoch_idx,
                scope=locals(),
                **{
                    "train__Loss": total_loss1 / len(dataloader.dataset),
                    "train__Accuracy": total_correct1 / len(dataloader.dataset),
                })

            print('Model2:Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f}'.format(epoch_idx,
                                                                                  args.epochs,
                                                                                  total_loss2 /
                                                                                  len(
                                                                                      dataloader.dataset),
                                                                                  total_correct2 / len(dataloader.dataset)))
            nsml.report(
                summary=True,
                step=epoch_idx,
                scope=locals(),
                **{
                    "train__Loss": total_loss2 / len(dataloader.dataset),
                    "train__Accuracy": total_correct2 / len(dataloader.dataset),
                })

            print('Model3:Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f}'.format(epoch_idx,
                                                                                  args.epochs,
                                                                                  total_loss3 /
                                                                                  len(
                                                                                      dataloader.dataset),
                                                                                  total_correct3 / len(dataloader.dataset)))
            nsml.report(
                summary=True,
                step=epoch_idx,
                scope=locals(),
                **{
                    "train__Loss": total_loss3 / len(dataloader.dataset),
                    "train__Accuracy": total_correct3 / len(dataloader.dataset),
                })
