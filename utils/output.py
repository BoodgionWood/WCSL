import csv
import os

import torch


def create_output_path(args, count=0, folder_name=None):
    if folder_name is None:
        if args.server:
            folder_name = args.method + "_" + args.dataset + "_N-" + str(args.num_pts) + "_n-" + str(
                args.num_samples) + "_p-" + str(args.data_dim) + "_reg-" + str(args.reg) + "_lr-" + str(args.lr)
        else:
            folder_name = args.dataset + "_lr-" + str(args.lr)
    output_path = args.output_path + folder_name

    if count == 0:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            output_path = output_path + "/"
        else:
            output_path = create_output_path(args, count + 1, folder_name)
    else:
        output_path = output_path + "_" + str(count)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            output_path = output_path + "/"
        else:
            output_path = create_output_path(args, count + 1, folder_name)

    return output_path


def write_to_csv(filename, row, header=None):
    with open(filename, 'a', newline='') as file:  # 'a' means append mode
        writer = csv.writer(file)
        # If the file is empty, write the header first
        if file.tell() == 0 and header is not None:
            writer.writerow(header)
        writer.writerow(row)


def save_checkpoint(state, filename):
    torch.save(state, os.path.join(filename,  'checkpoint.pt'))


def load_checkpoint(filename):
    try:
        checkpoint = torch.load(filename + '_checkpoint.pt')
        return checkpoint
    except FileNotFoundError:
        return None
