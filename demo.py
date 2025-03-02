import os
name_list = list(open('list/tad-clip-test.list'))
for i in name_list:
    i = i[:-1]
    src_file = i.replace('10crop_clip','sent_emb_n').replace('.npy','_emb.npy')
    dst_file = src_file.replace('sent_emb_n','test')
    os.system('cp {} {}'.format(src_file,dst_file))
    print(i)