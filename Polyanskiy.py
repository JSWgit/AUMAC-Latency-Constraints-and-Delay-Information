import torch as tor
from typing import Tuple 
import matplotlib.pyplot as plt

my_device=tor.device("cuda" if tor.cuda.is_available() else "cpu")

def cc(a,b):
    return tor.lgamma(a+1)-tor.lgamma(b+1)-tor.lgamma(a-b+1)
def test( snr, ka, rho: tor.Tensor, rho1: tor.Tensor,s: tor.Tensor):
    n=tor.tensor(38400)
    logm=128*tor.log(tor.tensor(2))
    p=10**(snr/10)
    R2=1/n*cc(ka,s)
    R1=1/n*logm-1/n/s*tor.lgamma(s+1)
    D=(p*s-1)**2+4*p*s*(1+rho*rho1)/(1+rho)
    lam=(p*s-1+pow(D,1/2))/(4*(1+rho*rho1)*p*s)
    mu=rho*lam/(1+2*p*s*lam)
    b=rho*lam-mu/(1+2*s*p*mu)
    a=rho/2*tor.log(1+2*p*s*lam)+0.5*tor.log(1+2*s*p*mu)
    E0=rho1*a+0.5*tor.log(1-2*b*rho1)
    E=-(-rho*rho1*s*R1-rho1*R2+E0)
    val, ind=E.min(dim=-1)
    return val, ind

def main():
    n=38400
    rho1=tor.arange(0,1+1e-2,1e-2)
    ka_range=tor.arange(20,301,20)
    ka_range=tor.cat([tor.tensor([1]),tor.tensor([10]),ka_range]).to(my_device)
    snr_range=tor.arange(-22,-10,1e-2, device=my_device)
    rho_save=tor.zeros(300)
    rho1_save=tor.zeros(300)
   
    err=tor.zeros(len(snr_range),len(ka_range))
    for i_ka, ka in enumerate(ka_range):
        val_save=tor.zeros(ka) # type: ignore
        for i_snr, snr in enumerate(snr_range):
            for s in range(1,ka+1): 
                print(f'[s is] {s}')
                temp=tor.tensor(float('inf'), dtype=tor.float64)
                for rho in tor.arange(0,1+1e-2,1e-2):
                    val, ind=test(snr, ka, rho,rho1, tor.tensor(s)) # type: ignore
                    if val<= temp:
                        temp=val
                        val_save[s-1]=temp
                err[i_snr,i_ka]+=(tor.exp(n*val_save[s-1])*s/ka).sum() # type: ignore
                if err[i_snr,i_ka]>=0.05:
                    break
            print (f'[ka,snr, err is] {ka},{snr}, {err[i_snr,i_ka]}')
            if err[i_snr,i_ka]<1e-6:
                break
        data = {
            'ka': ka_range.cpu(),
            'snr': snr_range.cpu(),
            'err':err.cpu(),
            'n': n
            }
        tor.save(data, 'Polyanskiy.pt') 


if __name__ == "__main__":
    main()
    


