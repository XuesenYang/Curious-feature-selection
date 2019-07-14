addpath(genpath(pwd));
name={'sonar','Hill','SPECTHeart','Libras Movement','LSVT'...
     'Urban land cover','ionosphere','colon','ForestTypes','GLIOMA','lung_discrete','Yale'};
addpath(genpath('dataset'));
num_dataset=length(name);
for id=1:num_dataset %:num_dataset   %选择数据集，可以手动改
dataset=name{id};
switch dataset    
    case 'sonar'
 load('sonar.mat')
    case 'Hill'
 load('Hill.mat')
    case 'SPECTHeart'
 load('SPECTHeart.mat')
    case 'Libras Movement'
 load('Libras Movement.mat')
    case 'LSVT'
 load('LSVT_voice_rehabilitation.mat')
    case 'Urban land cover'
 load('Urban land cover.mat')
    case 'ionosphere'
 load('ionosphere.mat')
    case 'colon'
 load('colon.mat')
    case 'ForestTypes'
 load('ForestTypes.mat')
    case 'GLIOMA'
 load('GLIOMA.mat')
    case 'lung_discrete'
 load('lung_discrete.mat')
    case 'Yale'
 load('Yale.mat')
end 
tic
maxrun=30;
fit=zeros(1,maxrun);
n_si=zeros(1,maxrun);
for run=1:maxrun
F=size(data,2)-1;
N_data=size(data,1);
n=F+1;
N_epi=100;
Gamma=0.01;
itr=round(10*N_data/N_epi);
threshold=0.1;
fitness=zeros(1,itr);
for i=1:itr
    rand_sample=randperm(N_data);
    x_epi=data(rand_sample(1:N_epi),:);
    rand_x_epi=randperm(N_epi);
    x_epi_train=x_epi(rand_x_epi(1:0.8*N_epi),:);
    x_epi_vali=x_epi(rand_x_epi(0.8*N_epi+1:end),:);
    F_sel=[];
    F_avail=[1:F];
    Q=ones(1,F);
    E=0.5*ones(1,F);
    e=zeros(1,F);
    r=zeros(1,F);
    
    if i<itr/4
        Epsilon=0.09;
    elseif i>=itr/4 && i<itr/2
        Epsilon=0.05;
    elseif i>=itr/2 && i<itr*3/4
        Epsilon=0.01;
    else
        Epsilon=0.005;
    end
    if i<itr/4
        Alpha=0.9;
    elseif i>=itr/4 && i<itr/2
        Alpha=0.5;
    elseif i>=itr/2 && i<itr*3/4
        Alpha=0.3;
    else
        Alpha=0.1;
    end
    
    while ~isempty(F_avail)
        if rand<Epsilon
            n_avail=length(F_avail);
            rand_n_avail=randperm(n_avail);
            action=F_avail(1,rand_n_avail(1));
        else
            if max(Q)<threshold 
                F_avail=[];
            else
               [~,action]=max(Q);
            end   
        end
        
        for j=1:length(F_avail)
            train_data=x_epi_train(:,[F_sel,F_avail(j),end]);
            vali_data=x_epi_vali(:,[F_sel,F_avail(j),end]);
            mdl = fitcknn(train_data(:,1:end-1),train_data(:,end),'NumNeighbors',4,'Standardize',1);
            Acl=predict(mdl,vali_data(:,1:end-1));
            e(1,F_avail(j))=sum(Acl~=vali_data(:,end))/size(vali_data,1);
            r(1,F_avail(j))=E(1,F_avail(j))-e(1,F_avail(j));
            Q(1,F_avail(j))=Q(1,F_avail(j))+Alpha*(r(1,F_avail(j))+Gamma*max(Q)-Q(1,F_avail(j)));
        end
        E=e;
        if ~isempty(F_avail)
            F_sel=[F_sel,action];
        else
            F_sel=F_sel;
        end
        Q(1,action)=-inf;
        F_avail(F_avail==action)=[];
    end
        fitness(1,i)=func(data,F_sel);
        n_size(1,i)=length(F_sel);   
end
[fit(1,run),min_id]=min(fitness);
n_si(1,run)=n_size(1,min_id);
end
best_err=mean(fit);
best_f=mean(n_si);
t=toc;
file_name=['CFS+',dataset];
save(file_name,'best_err','best_f','t','fit');
end


