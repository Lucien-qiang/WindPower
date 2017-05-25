import os
import scipy.io as sio
g=open('predictt','w')
for fl1 in os.listdir('PredictData'):
    for fl2 in os.listdir('PredictData/'+fl1):
        f=open('PredictData/'+fl1+'/'+fl2+'/'+'T_out_ncl.txt')
        i=0
        for line in f:
            g.write(fl1+fl2+'\t'+str(i)+'\t'+line)
            i=i+1
g.close()