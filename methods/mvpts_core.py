import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm

from loss.sinkhorn import SinkhornLoss
from dataset.utils import sample_from_loader, pack_data_to_loader
from loss.loss import compute_M
from utils.output import save_checkpoint, load_checkpoint


def mvpts(args, train_loader, start_time=None, device="cuda:0", filename=None):
    args.method = "mvpts"
    lossfn = SinkhornLoss(reg=args.reg, max_iter=10000)

    if args.train_round > 0:
        checkpoint = load_checkpoint(args.model_path + args.unique_name)
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        samples = checkpoint['samples'].to(device)  # Load the tensor and ensure it's on the correct device
        samples.requires_grad = True
        optimizer = optim.Adam([samples], lr=args.lr)
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.num_epochs, steps_per_epoch=1)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        loss_record = checkpoint['loss_record']
        start_epoch = checkpoint['epoch']
    else:
        samples = sample_from_loader(train_loader, args.num_samples).to(device)
        samples.requires_grad = True
        optimizer = optim.Adam([samples], lr=args.lr)
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.num_epochs, steps_per_epoch=1)
        loss_record = []
        start_epoch = 0

    iter_start_time = time.time()  # To sum up the time taken for each iteration

    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        loss_total = 0
        samples_before_update = samples.clone().detach()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            M = compute_M(data, samples, args.m_distance)
            loss = lossfn(M)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        # scheduler.step()
        loss_record.append(loss_total / len(train_loader))
        # use samples_diff_norm as convergence criteria
        # samples_diff = samples.clone().detach() - samples_before_update
        # samples_diff_norm = torch.norm(samples_diff, p=2)
        # Use sinkhorn loss difference as convergence criteria

        # plot_three_scatters(samples, samples_before_update, x_r=None, y_r=None,filename=args.plot_path + str(epoch))

        # Save checkpoint at the end of each epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'samples': samples,
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'loss_record': loss_record,
        }, args.model_path+args.unique_name)

        if args.mvpts_thred > 0:
            M1 = compute_M(samples.clone().detach(), samples_before_update, args.m_distance)
            # M2 = compute_M(samples_before_update, samples_before_update, args.m_distance)
            # convergence_metric = abs(lossfn(M1) - lossfn(M2))
            convergence_metric = lossfn(M1)
            print(f'Epoch: {epoch}, Loss: {loss_total / len(train_loader)}, Convergence Metric: {convergence_metric}')
            if convergence_metric < args.mvpts_thred:
                # Save the number of iteration and average time per iteration to csv
                iter_end_time = time.time()
                iter_time = (iter_end_time - iter_start_time) / (epoch + 1)
                with open(filename, 'a') as f:
                    # write args.num_pts, args.num_samples, args.data_dim, convergence criteria, iteration_count, iter_time to csv
                    f.write(
                        f'{args.num_pts},{args.num_samples},{args.data_dim},{args.mvpts_thred},{epoch + 1},{iter_time}\n')
                break
        else:
            print(f'Epoch: {epoch}, Loss: {loss_total / len(train_loader)}')

        # Time check to end training if it exceeds 3.5 hours
        if start_time is not None:
            if time.time() - start_time > 12600:
                return False

    # Save samples
    mvpts_loader = pack_data_to_loader(samples, batch_size=args.sample_batch_size)
    torch.save(mvpts_loader, os.path.join(args.model_path + args.unique_name , "mvpts_loader.pt") )
    if start_time is None:
        return samples
    else:
        return True


if __name__ == '__main__':
    pass