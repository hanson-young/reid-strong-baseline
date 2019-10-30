import os
import numpy.random as npr

if __name__ == '__main__':
    root_path = "/media/yj_data/DataSet/reid/kesci"
    test_file = os.path.join(root_path, "test_list.txt")
    query_file = os.path.join(root_path, "query_list.txt")
    gallery_file = os.path.join(root_path, "gallery_list.txt")

    test_total_lines = []
    with open(test_file, 'r') as f:
        test_total_lines = f.readlines()

    qf = open(query_file, "w")
    gf = open(gallery_file, "w")

    pids_dict = {str(i):list() for i in range(4000, 5000)}
    for line in test_total_lines:
        img_path, pid = line.strip().split()
        # pids[int(pid)] += 1
        tmp = pids_dict[pid]
        tmp.append(line)
        pids_dict[pid] = tmp

    for k, v in pids_dict.items():
        if len(v) >= 2:
            qf.write(v[0])
            for l in v[1:]:
                gf.write(l)

    qf.close()
    gf.close()

