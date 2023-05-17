function [TPR,FPR] = rate(gnd_truth,predictions)
    conf  = confusionmat(gnd_truth,predictions);
    TPR = conf(1,1)/(conf(1,1)+conf(2,1));
    FPR = conf(1,2)/(conf(1,2)+conf(2,2));
    
end