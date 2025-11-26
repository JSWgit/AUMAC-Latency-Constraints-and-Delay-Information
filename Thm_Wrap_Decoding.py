# this code is for AUMAC TIT paper, single antenna, analyzed with chernoff bound with Gallager's rho trick
# the program simulate the PUPE of an AUMAC using wrap-decoder
import torch as tor
from typing import Tuple 
import matplotlib.pyplot as plt
import mpmath
import time
mpmath.mp.dps = 100 

def logcc_large(a,b):
    return  float(mpmath.loggamma(a + 1) - mpmath.loggamma(b + 1) - mpmath.loggamma(a - b + 1))

def cc(a,b):
    return tor.lgamma(a+1)-tor.lgamma(b+1)-tor.lgamma(a-b+1)

def err(rho, rho1, s, ka, snr,dm,n,my_device):
    part0_val_temp=0
    part0_thm6_val_temp=0
    m=128
    t=tor.arange(1e-4, min(1/(4*rho), 3),0.01, device=my_device) # type: ignore 
    t_u,rho1_u=tor.meshgrid(t,rho1,indexing='ij')
    p=10**(snr/10)
    theta=tor.tensor(logcc_large(2**m-int(ka),int(s))+tor.lgamma(s+1))
    theta_thm6=tor.tensor(logcc_large(2**m-int(ka),int(s))+s*tor.log(1+dm))
    theta=theta.to(my_device)
    theta_thm6=theta_thm6.to(my_device)
    theta0=(rho*rho1_u*theta+rho1_u*cc(ka,s)).to(my_device)
    theta0_thm6=(rho*rho1_u*theta_thm6+rho1_u*cc(ka,s)).to(my_device)
    logmu=(1-rho)*n/2*tor.log(1+2*s*p*t_u)-n/2*tor.log(rho*t_u)
    As=(1+2*s*p*t_u*(1+rho))/(rho*t_u)
    C4=(1+2*s*p*t_u*(1+rho)*(1-4*rho*rho1_u*t_u))/(1+2*s*p*t_u*(1+rho))
    C2=(1+2*s*p*t_u*(1+rho)*(1-2*rho*rho1_u*t_u))/(1+2*s*p*t_u*(1+rho))
    fq0=n*rho1_u/2*tor.log(As)+dm/2*tor.log(C4)+(n-dm)/2*tor.log(C2)
    fq1_part1=-n/2*tor.log(rho*t_u)+dm/2*tor.log(1+2*s*p*t_u*(1+rho)*(1-4*rho*t_u))+(n-dm)/2*tor.log(1+2*s*p*t_u*(1+rho)*(1-2*rho*t_u))
    fq1_part2=dm*rho*t_u*(1-4*rho*t_u)/(1+2*s*p*t_u*(1+rho)*(1-4*rho*t_u))+(n-dm)*rho*t_u*(1-2*rho*t_u)/(1+2*s*p*t_u*(1+rho)*(1-2*rho*t_u))
    part0_val, part0_ind=tor.exp(theta0+rho1_u*logmu-fq0).min(dim=0)
    part0_thm6_val, part0_thm6_ind=tor.exp(theta0_thm6+rho1_u*logmu-fq0).min(dim=0)
    for s1 in tor.arange(1,ka-s+1):
        theta0_s1=(rho*1*theta+1*cc(ka,s)).to(my_device)
        theta0_thm6_s1=(rho*1*theta_thm6+1*cc(ka,s)).to(my_device)

        barlam=4*s1*p*(1-2*rho*t_u)*rho*t_u/(1+2*s*p*t_u*(1+rho)*(1-2*rho*t_u))
        fq1=fq1_part1+2*s1*p*fq1_part2*(1+2*barlam+barlam**2/3)**(-0.5)
        theta_s1=1*(cc(ka-s,s1)+tor.lgamma(s+s1+1)-tor.lgamma(s+1))
        theta_s1_thm6=1*(cc(ka-s,s1)+s1*tor.log(dm))
        part0_s1_val, part0_s1_ind=tor.exp(theta0_s1+theta_s1+1*logmu-1*fq1).min(dim=0)
        part0_s1_thm6_val, part0_s1_thm6_ind=tor.exp(theta0_thm6_s1+theta_s1_thm6+1*logmu-1*fq1).min(dim=0)
        part0_val_temp+=part0_s1_val
        part0_thm6_val_temp+=part0_s1_thm6_val

    part0_val_temp=part0_val_temp**(rho1)
    part0_thm6_val_temp=part0_thm6_val_temp**(rho1)

    err_final=(part0_val+part0_val_temp)
    err_final_thm6=(part0_thm6_val+part0_thm6_val_temp)

    err_final_val, err_final_ind=err_final.min(dim=-1)
    err_final_thm6_val, err_final_thm6_ind=err_final_thm6.min(dim=-1)

    return err_final_val,err_final_thm6_val

    
def main():
    my_device=tor.device("cuda" if tor.cuda.is_available() else "cpu")
    rho1=tor.arange(1e-2,1+1e-2,1e-2,device=my_device)
    rho1=tor.cat([tor.tensor([1e-9],device=my_device),rho1])
    alp=0.2
    n=tor.tensor(38400)
    ka_range=tor.arange(20,301,20)
    ka_range=tor.cat([tor.tensor([1]),tor.tensor([10]),ka_range]).to(my_device)
    snr_range=tor.arange(-22,-10,1e-2, device=my_device)
    dm=alp*n
    rho_range=tor.arange(1e-2,1+1e-2,1e-2, device=my_device)
    rho_range=tor.cat([tor.tensor([1e-9],device=my_device),rho_range])
    err_record=tor.zeros(len(ka_range), len(snr_range),device=my_device)
    err_record_thm6=tor.zeros(len(ka_range), len(snr_range),device=my_device)
    start=0
    start_temp=0
    skip2=0
    for i_ka, ka in enumerate(ka_range):
        start_cond=0
        skip2=0
        start=start_temp
        print(f'[now we are in ] {ka}')
        s_range = tor.cat((
            ka[None],                                  # [ka]
            ka.new_tensor([1]),
            tor.arange(2, ka, device=my_device).flip(0) # type: ignore
        )) if ka > 2 else ka.new_tensor([1])
        for i_snr, snr in enumerate(snr_range[start:]):
            for i_s, s in enumerate(s_range):
                err_temp=tor.zeros(len(rho_range),device=my_device)
                err_temp_thm6=tor.zeros(len(rho_range),device=my_device)
                for i_rho, rho in enumerate(rho_range):
                    err_temp[i_rho],err_temp_thm6[i_rho]=err(rho, rho1, s, ka, snr,dm,n,my_device)
                err_record[i_ka, start+i_snr]+=s/ka*tor.min(err_temp)
                err_record_thm6[i_ka, start+i_snr]+=s/ka*tor.min(err_temp_thm6)
                print(f'[err_thm5, err_thm6, s, snr] are [{err_record[i_ka, start+i_snr]},{err_record_thm6[i_ka, start+ i_snr]}, {s}, {snr}]')
                if err_record[i_ka, start+i_snr]>0.05:
                    break
                if skip2==1 and err_record_thm6[i_ka, start+i_snr]>0.05:
                    print(f'{start_cond}, and, {err_record_thm6[i_ka, start+i_snr]}')
                    break
            if err_record[i_ka, start+i_snr]<0.05 and i_s==ka-1: # type: ignore
                skip2=1
            if err_record[i_ka, start+i_snr]<1 and start_cond==0 and i_s==ka-1: # type: ignore
                start_cond=1
                start_temp=start+i_snr# type: ignore
            if err_record_thm6[i_ka, start+i_snr]<1e-5:
                break
    data = {
    'ka': ka_range.cpu(),
    'snr': snr_range.cpu(),
    'err':err_record.cpu(),
    'err_thm6':err_record_thm6.cpu(),
    'alp':alp,
    'n': n
    }
    # tor.save(data, 'my_data_AUMAC_Chernoff_thm2.pt')    
if __name__ == "__main__":
    main()
    

