%load
close all
clc
t=time;
%set params
maxDelta=0.001*ones(1,7) %movement threshold for when the robot is considered static, default=0.001
minS=3  %minimal time[s] the robot needs to stand still to record the point, default=3
minCount=100*minS;
%filter data
Wn = 5/100; % Normalized cutoff frequency
[Fb,Fa]=butter(4,Wn);
qDot=filter(Fb,Fa,qDotOut);
q=filter(Fb,Fa,qOut);
%%
%find stopped points
clear idx
found=0;
for idx=1:length(t)
    if abs(mean(qDot(idx,:)))<maxDelta
        if idx==1
            count(idx)=1;
        else
            count(idx)=count(idx-1)+1;
        end
    else
        count(idx)=0;
    end
    
end
count(idx+1)=0;
pts=zeros(1,7);
%isolate maximums
for idx=minCount:length(t)+1
    if abs(count(idx)-count(idx-1))>minCount-1
        found=found+1;
        idxFound(found)=idx-1;
        pts(found,:)=q(idx-1,:);
    end
end
%plot found points
figure()
countVec=minCount*ones(length(t),1);
plot(t,count([1:length(count)-1]),t,countVec,t(idxFound),count(idxFound),'o');
legend('count','threshold','found Points');
pts
save("recorded2.mat","pts");
% for ii=1:size(pts,1)
%     for jj=1:size(pts,2)
%         
%     end
% end
%%
%plot
figure();
subplot(2,1,1);
plot(t,q);
hold on
for jj=1:found
    xline(idxFound(jj)/100);
end
hold off
title("position");
legend('q1','q2','q3','q4','q5','q6','q7');
xlabel("\v{c}as [s]", 'Interpreter','latex')
ylabel("kot [rad]", 'Interpreter','latex')
subplot(2,1,2);
plot(t,qDot);
hold on
for jj=1:found
    xline(idxFound(jj)/100);
end
hold off
title("velocity");
legend('q1','q2','q3','q4','q5','q6','q7');
xlabel("\v{c}as [s]", 'Interpreter','latex')
ylabel("kotna hitrost [rad/s]", 'Interpreter','latex')
text=strcat("position and velocity");
sgtitle(text,'FontSize',14);