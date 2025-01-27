# written by JhihYang Wu <jhihyangwu@arizona.edu>
# generates a csv file when there are a lot of eval out folders
# to run simply execute: python eval_gen_csv.py --path path_to_folder_with_eval_out_folders

import os
import csv
import click

@click.command()
@click.option("--path", help="Which path has a bunch of eval out folders", required=True)

def main(**kwargs):
    out_dir = kwargs["path"]

    with open("all.csv", "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["directory", "fid", "lpips", "dists", "psnr", "ssim"])
        for path in sorted(os.listdir(out_dir)):
            if path.endswith("tmp_fid"):
                continue  # not a result directory so skip
            path = os.path.join(out_dir, path)  # get full path
            path1 = os.path.join(path, "finish_2.txt")
            path2 = os.path.join(path, "finish_3.txt")
            if os.path.exists(path1) and os.path.exists(path2):
                with open(path1, "r") as file:
                    final_line = file.readlines()[-1]
                    assert final_line.startswith("final psnr: ")
                    _, _, psnr, _, _, ssim, _, _, lpips, _, _, dists = final_line.split()
                with open(path2, "r") as file:
                    final_line = file.readlines()[-1]
                    assert final_line.startswith("FID: ")
                    fid = final_line.split()[-1]
                csv_writer.writerow([path, fid, lpips, dists, psnr, ssim])

if __name__ == "__main__":
    main()
