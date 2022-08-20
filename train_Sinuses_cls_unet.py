import argparse
import os
import pathlib
from pyexpat import model
import time
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from monai.data import decollate_batch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from config_OPC import get_config
from dataset.brats import get_datasets
from loss import EDiceLoss
from loss.dice_opc import EDiceLoss_Val
from loss.criterions import CoxLoss,CIndex_lifeline,cox_log_rank,idh_cross_entropy
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
    dice_metric_batch
# from vtunet.vision_transformer_OPC_MTL_qian import VTUNet as ViT_seg, Surv_network_qian_unet

from cnn.unet3d import UNet3D, Surv_network_qian_unet

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,accuracy_score

from dataset.OPC_seg_surv_data import OPCData

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='VTUNET BRATS 2021 Training')
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--val', default=1, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/vt_unet_base.yaml", metavar="FILE",
                    help='path to config file', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')


def main(args):
    # setup
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")

    args.exp_name = "logs_bidou_2022-5-24-unet-cls"
    args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)
    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    args.checkpoint_folder = pathlib.Path(f"./runs/{args.exp_name}/model_1")

    # Create model
    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    config = get_config(args)
    # model_1 = ViT_seg(config, num_classes=args.num_classes,
    #                   embed_dim=yaml_cfg.get("MODEL").get("SWIN").get("EMBED_DIM"),
    #                   win_size=yaml_cfg.get("MODEL").get("SWIN").get("WINDOW_SIZE")).cuda()
    # model_1.load_from(config)

    model_1 = UNet3D()

    Surv_model = Surv_network_qian_unet()
    Surv_model = Surv_model.cuda()

    if args.resume:
        args.checkpoint = args.checkpoint_folder / "model_best.pth.tar"
        reload_ckpt_bis(args.checkpoint, model_1)

    print(f"total number of trainable parameters {count_parameters(model_1)}")

    model_1 = model_1.cuda()

    model_file = args.save_folder_1 / "model.txt"
    with model_file.open("w") as f:
        print(model_1, file=f)

    nets = {
        'seg': model_1,
        'surv': Surv_model
    }


    criterion = EDiceLoss().cuda()
    criterian_val = EDiceLoss_Val().cuda()
    criterian_dice = criterian_val.binary_dice
    metric = criterian_val.metric
    print(metric)
    #params = model_1.parameters()

    params = [p for v in nets.values() for p in list(v.parameters())]

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay) # 定义优化器

    # full_train_dataset, l_val_dataset, bench_dataset = get_datasets(args.seed, fold_number=args.fold)

    full_train_dataset = OPCData(list_file='train.csv',mode='train') # 训练集
    l_val_dataset = OPCData(list_file='test_label_3.csv',mode='valid')# 验证集
    l_test_dataset = OPCData(list_file='test_label_3.csv',mode='valid')# 测试集


    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(l_test_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    # bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)
 


    print("Train dataset number of batch:", len(train_loader))
    print("Val dataset number of batch:", len(val_loader))
    # print("Bench Test dataset number of batch:", len(bench_loader))

    # Actual Train loop
    best_1 = 0.0
    patients_perf = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print("start training now!")

    for epoch in range(args.epochs):
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()

            # Setup
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_ = AverageMeter('Loss', ':.4e')

            mode = "train" if model_1.training else "val"
            batch_per_epoch = len(train_loader)
            progress = ProgressMeter(
                batch_per_epoch,
                [batch_time, data_time, losses_],
                prefix=f"{mode} Epoch: [{epoch}]")

            end = time.perf_counter()
            metrics = []

            nets['seg'].train()
            nets['surv'].train()

            for i, batch in enumerate(zip(train_loader)):
                torch.cuda.empty_cache()
                # measure data loading time
                data_time.update(time.perf_counter() - end)

                inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["seg"].float()

                label = batch[0]['label']

                label = Variable(label)
                label = label.cuda()


                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()

                optimizer.zero_grad()

                segs_S1,Trans_features = nets['seg'](inputs_S1)

                #print('bottle_features的shape：',Trans_features.shape)

                surv_out = nets['surv'](Trans_features)


                loss_ = criterion(segs_S1, labels_S1)
                loss_surv = idh_cross_entropy(surv_out,label)

                loss = loss_ + loss_surv

                t_writer_1.add_scalar(f"Loss/{mode}{''}",
                                      loss_.item(),
                                      global_step=batch_per_epoch * epoch + i)

                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                else:
                    print("NaN in model loss!!")

                # compute gradient and do SGD step
                # loss_.backward() #原代码
                loss.backward()
                optimizer.step()

                t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                      global_step=epoch * batch_per_epoch + i)

                if scheduler is not None:
                    scheduler.step()

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                # Display progress
                progress.display(i)

            t_writer_1.add_scalar(f"SummaryLoss/train", losses_.avg, epoch)

            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")
            torch.cuda.empty_cache()

            # Validate at the end of epoch every val step

            print('训练完{}轮了，开始验证......'.format(epoch))

            risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])

            idh_probs = []
            idh_class = []
            idh_target = []
            idh_preds = []



            nets['seg'].eval()
            nets['surv'].eval()

            if (epoch + 1) % args.val == 0:

                #validation_dices = []

                validation_dice = 0.0

                with torch.no_grad():
                    for i, batch in enumerate(zip(val_loader)):
                        data_time.update(time.perf_counter() - end)

                        inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["seg"].float()

                        label = batch[0]['label']

                        label = Variable(label)
                        label = label.cuda()

                        inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                        inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()

                        segs_S1,Trans_features = nets['seg'](inputs_S1)
                        surv_out = nets['surv'](Trans_features)
                        

                        validation_dice_sample = criterian_dice(segs_S1,labels_S1)
                        # validation_dices.append(validation_dice)

                        idh_pred = F.softmax(surv_out, 1)
                        idh_pred_class = torch.argmax(idh_pred, dim=1)

                        #print(idh_pred)

                        #print(idh_pred[0][1])


                
                        idh_probs.append(idh_pred[0][1].cpu().numpy())
                        idh_class.append(idh_pred_class.item())
                        idh_target.append(label.item())

                        idh_preds.append(idh_pred.cpu().numpy()[0])
        


                        print(validation_dice_sample)

                        validation_dice += validation_dice_sample / len(val_loader)

                    accuracy = accuracy_score(idh_target,idh_class)

                    print('准确率：',accuracy)
                    print("********************************************************************************************")

                    auc = roc_auc_score(idh_target,idh_probs)
                    print('####################')
                    print('测试集的roc为：',auc)
                    print('####################')


                    # guess = idh_class
                    # fact = idh_target
                    # classes = list(set(fact))
                    # classes.sort()
                    # confusion = confusion_matrix(guess, fact)
                    # plt.imshow(confusion, cmap=plt.cm.Blues)
                    # indices = range(len(confusion))
                    # plt.xticks(indices, classes)
                    # plt.yticks(indices, classes)
                    # plt.colorbar()
                    # plt.xlabel('predict')
                    # plt.ylabel('Truth')
                    # for first_index in range(len(confusion)):
                    #     for second_index in range(len(confusion[first_index])):
                    #         plt.text(first_index, second_index, confusion[first_index][second_index])

                    # # plt.savefig('分期混淆矩阵_train_all_T1_train_stage/混淆矩阵-分期-T1-deep_{}.pdf'.format(epoch))
 
                    # plt.show()
                    # plt.clf()




                #validation_loss_1, validation_dice = step(val_loader, model_1, Surv_model,criterian_val, metric, epoch, t_writer_1,
                #                                          save_folder=args.save_folder_1,
                #                                          patients_perf=patients_perf)

                t_writer_1.add_scalar(f"SummaryDice", validation_dice, epoch)

                print('当前轮次的验证dice：',validation_dice)



                t_writer_1.add_scalar(f"auc test", auc, epoch)
                ## t_writer_1.add_scalar(f"p value val", pvalue_test, epoch)


                if validation_dice > best_1:
                    print(f"Saving the model with DSC {validation_dice}")
                    best_1 = validation_dice

                    final_name = os.path.join(args.save_folder_1, 'model_epoch_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'en_state_dict': nets['seg'].state_dict(),
                        'seg_state_dict': nets['surv'].state_dict(),
                        'optim_dict': optimizer.state_dict(),
                        'scheduler_dict': scheduler.state_dict()
                    },final_name)

                    # model_dict = nets['seg'].state_dict()
                    # save_checkpoint(
                    #     dict(
                    #         epoch=epoch,
                    #         state_dict=model_dict,
                    #         optimizer=optimizer.state_dict(),
                    #         scheduler=scheduler.state_dict(),
                    #     ),
                    #     save_folder=args.save_folder_1, )

                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s")
                torch.cuda.empty_cache()

            # print('训练完{}轮了，开始测试......'.format(epoch))

            # risk_pred_all_test, censor_all_test, survtime_all_test = np.array([]), np.array([]), np.array([])

            # nets['seg'].eval()
            # nets['surv'].eval()

            # test_dice = 0.0

            # with torch.no_grad():
            #     for i, batch in enumerate(zip(test_loader)):
            #         data_time.update(time.perf_counter() - end)

            #         inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["seg"].float()

            #         label = batch[0]['label']

            #         label = Variable(label)
            #         label = label.cuda()

            #         inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
            #         inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()

            #         segs_S1,Trans_features = nets['seg'](inputs_S1)
            #         surv_out = nets['surv'](Trans_features)

            #         test_dice_sample = criterian_dice(segs_S1,labels_S1)
    
            #         print(test_dice_sample)

            #         test_dice += test_dice_sample / len(test_loader)

            #         risk_pred_all_test = np.concatenate((risk_pred_all_test, surv_out.detach().cpu().numpy().reshape(-1)))   # Logging Information
            #         censor_all_test = np.concatenate((censor_all_test, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
            #         survtime_all_test = np.concatenate((survtime_all_test, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information


               

            # t_writer_1.add_scalar(f"SummaryDice_test", test_dice, epoch)

            # print('当前轮次的测试dice：',test_dice)

            # print('*#*#*#*#*# 生存结果评估#*#*#*#*#*#*#*#*#*#*#*##*#*#*#*#*#*##*')

            # cindex_test = CIndex_lifeline(risk_pred_all_test, 1-censor_all_test, survtime_all_test)
            # pvalue_test = cox_log_rank(risk_pred_all_test, 1-censor_all_test, survtime_all_test)

            # print('生存分析C-Index：',cindex_test)
            # print('生存分析p值：',pvalue_test)


            # t_writer_1.add_scalar(f"C-Index test", cindex_test, epoch)
            # t_writer_1.add_scalar(f"p value test", pvalue_test, epoch)






        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            break


def step(data_loader, model,model2, criterion: EDiceLoss_Val, metric, epoch, writer, save_folder=None, patients_perf=None):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    mode = "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []

    for i, val_data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        patient_id = val_data["ids"]

        model.eval()
        model2.eval()
        with torch.no_grad():
            val_inputs, val_labels, val_survival_months, val_censor = (
                val_data["image"].cuda(),
                val_data["seg"].cuda(),
                val_data["survival_months"].cuda(),
                val_data["censorship"].cuda(),
            )
            val_outputs = inference(val_inputs, model)
            val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs)]

            segs = val_outputs
            targets = val_labels
            loss_ = criterion(segs, targets)
            dice_metric(y_pred=val_outputs_1, y=val_labels)

        if patients_perf is not None:
            patients_perf.append(
                dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
            )

        writer.add_scalar(f"Loss/{mode}{''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)

        # measure accuracy and record loss_
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            print("NaN in model loss!!")

        metric_ = metric(segs, targets)
        metrics.extend(metric_)

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    save_metrics(epoch, metrics, writer, epoch, False, save_folder)
    writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    dice_values = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()

    return losses.avg, dice_values


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
