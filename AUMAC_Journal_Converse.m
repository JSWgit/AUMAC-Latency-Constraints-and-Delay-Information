clear;
close all;
logM=100*log(2);
Ka=300:-3:3;
Ka=[Ka,1];
Ka=Ka(end:-1:1);

n=38400;
alpha=0.2;
dm=alpha *n;
pdb=-29:1e-2:0;
p=10.^(pdb./10);
pe=5e-2;
hb=pe*log2(1/pe)+(1-pe)*log2(1/(1-pe));
for k=1:length(Ka)
    
    ka=Ka(k);

    temp=((n-dm)/2*log2(1+ka*p)+dm*log2(1+(ka)/2*p))/ka-(1-pe)*log2(exp(logM)/ka)+hb;
    ind1(k)=find(temp>=0,1);
    temp=(n/2*log2(1+ka*p))/ka-(1-pe)*log2(exp(logM)/ka)+hb;
    ind2(k)=find(temp>=0,1);
    temp=((n+dm)/2*log2(1+ka*p))/ka-(1-pe)*log2(exp(logM)/ka)+hb;
    ind3(k)=find(temp>=0,1);
end


ebno1=log10(p(ind1)*n/2/(logM/log(2)))*10;
ebno2=log10(p(ind2)*n/2/(logM/log(2)))*10;
ebno3=log10(p(ind3)*n/2/(logM/log(2)))*10;






