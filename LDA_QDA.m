
N = 100;
x = linspace(5,14,N);
y = linspace(5,16,N);
[X,Y]=meshgrid(x,y)
h = (16-5)/N;

z = zeros(N^2,2);

for i = 1:N
    for k = 1:N
        z(N*(k-1)+i,1) = y(i);
    end
end

for i = 1:N
    for k = 1:N
        z(N*(k-1)+i,2) = y(k);
    end
end

N1=180;
mu = [8 11];
sigma = [1 1; 1 2];
R = chol(sigma);
g1 = repmat(mu,100,1) + randn(100,2)*R;

r1 = 8 + (13-8).*rand(80,1);
r2 = 6 + (12-6).*rand(80,1);
g2=[r1 r2];



All_sample=[g1;g2];


All_sample_label=zeros(size(All_sample(:,1)));
for k=1:length(All_sample)
    if k<=length(g1)
        All_sample_label(k)=1;
    else
        All_sample_label(k)=2;
    end
end


LDA_sample=[All_sample_label,All_sample,All_sample(:,1).^2,All_sample(:,1).*All_sample(:,2),All_sample(:,2).^2];


pi1=size(g1)/size(All_sample(:,1));
pi2=size(g2)/size(All_sample(:,1));


row_idx = (LDA_sample(:, 1) == 1);
LDA_sample_g1=LDA_sample(row_idx, 2:1:6);
mu1=mean(LDA_sample_g1);

row_idx = (LDA_sample(:, 1) == 2);
LDA_sample_g2=LDA_sample(row_idx, 2:1:6);
mu2=mean(LDA_sample_g2);

cov1=(LDA_sample_g1-mu1).'*(LDA_sample_g1-mu1);
cov2=(LDA_sample_g2-mu2).'*(LDA_sample_g2-mu2);

cov=(cov1+cov2)/(N1-2);

delta1_LDA=zeros(N1,1);
delta2_LDA=zeros(N1,1);
delta_classification_LDA=zeros(N1,1);


for k=1:length(All_sample)
    delta1_LDA(k)=log(pi1)+LDA_sample(k,2:1:6)*inv(cov)*transpose(mu1)-0.5*mu1*inv(cov)*transpose(mu1);
    delta2_LDA(k)=log(pi2)+LDA_sample(k,2:1:6)*inv(cov)*transpose(mu2)-0.5*mu2*inv(cov)*transpose(mu2);
    if delta1_LDA(k)>delta2_LDA(k)
        delta_classification_LDA(k)=1;
    else
        delta_classification_LDA(k)=2;
    end
end

Classification_Error_LDA=sum(abs(All_sample_label-delta_classification_LDA))/N1
QDA_sample=[All_sample_label,All_sample];


row_idx = (QDA_sample(:, 1) == 1);
QDA_sample_g1=QDA_sample(row_idx, 2:1:3);
Qmu1=mean(QDA_sample_g1);

row_idx = (QDA_sample(:, 1) == 2);
QDA_sample_g2=QDA_sample(row_idx, 2:1:3);
Qmu2=mean(QDA_sample_g2);

Qcov1=(QDA_sample_g1-Qmu1).'*(QDA_sample_g1-Qmu1);
Qcov2=(QDA_sample_g2-Qmu2).'*(QDA_sample_g2-Qmu2);

Qcov=(Qcov1+Qcov2)/(N1-2);

delta1_QDA=zeros(N1,1);
delta2_QDA=zeros(N1,1);
delta_classification_QDA=zeros(N1,1);
Qcovar1=((QDA_sample_g1-Qmu1).'*(QDA_sample_g1-Qmu1))/(length(g1)-1);
Qcovar2=((QDA_sample_g2-Qmu2).'*(QDA_sample_g2-Qmu2))/(length(g2)-1);
for k=1:length(All_sample)
    delta1_QDA(k)=log(pi1)-0.5*log(abs(det(Qcovar1)))-0.5*(Qmu1-QDA_sample(k,2:1:3))*inv(Qcov)*transpose(Qmu1-QDA_sample(k,2:1:3));
    delta2_QDA(k)=log(pi2)-0.5*log(abs(det(Qcovar2)))-0.5*(Qmu2-QDA_sample(k,2:1:3))*inv(Qcov)*transpose(Qmu2-QDA_sample(k,2:1:3));
    if delta1_QDA(k)>delta2_QDA(k)
        delta_classification_QDA(k)=1;
    else
        delta_classification_QDA(k)=2;
    end
end

Classification_Error_QDA=sum(abs(All_sample_label-delta_classification_QDA))/N1
x=linspace(5,14,100);
y=linspace(5,16,100);
[X,Y]=meshgrid(x,y);

LDA_z=[z,z(:,1).^2,z(:,1).*z(:,2),z(:,2).^2];

for k=1:10000
    delta1_LDAz(k)=log(pi1)+LDA_z(k,1:5)*inv(cov)*transpose(mu1)-0.5*mu1*inv(cov)*transpose(mu1);
    delta2_LDAz(k)=log(pi2)+LDA_z(k,1:5)*inv(cov)*transpose(mu2)-0.5*mu2*inv(cov)*transpose(mu2);
    if delta1_LDAz(k)>delta2_LDAz(k)
        delta_classification_LDAz(k)=1;
    else
        delta_classification_LDAz(k)=2;
    end
end
QDA_z=z;

for k=1:10000
    delta1_QDAz(k)=log(pi1)-0.5*log(abs(det(Qcovar1)))-0.5*(Qmu1-QDA_z(k,1:2))*inv(Qcov)*transpose(Qmu1-QDA_z(k,1:2));
    delta2_QDAz(k)=log(pi2)-0.5*log(abs(det(Qcovar2)))-0.5*(Qmu2-QDA_z(k,1:2))*inv(Qcov)*transpose(Qmu2-QDA_z(k,1:2));
    if delta1_QDAz(k)>delta2_QDAz(k)
        delta_classification_QDAz(k)=1;
    else
        delta_classification_QDAz(k)=2;
    end
end

figure(1)
h1=plot(g1(:,1),g1(:,2),'o'); hold on
h2=plot(g2(:,1),g2(:,2),'o')
contour(X,Y,reshape(delta_classification_LDAz,[100,100]))
legend('Class 1','Class 2','Location','best')
title(['Linear Discriminant Analysis - Misclassification Error ' num2str(Classification_Error_LDA)])

figure(2)
g1=plot(g1(:,1),g1(:,2),'o'); hold on
g2=plot(g2(:,1),g2(:,2),'o')
contour(Y,X,reshape(delta_classification_QDAz,[100,100]))
legend('Class 1','Class 2','Location','best')
title(['Quadratic Discriminant Analysis - Misclassification Error ' num2str(Classification_Error_QDA)])