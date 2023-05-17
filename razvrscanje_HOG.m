function [predictions,gnd_truth] = razvrscanje_HOG(D_train, D_test)
    

    predictions = zeros(1,length(D_test));
    gnd_truth = zeros(1,length(D_test));

    for i=1:length(D_test)
        d = zeros(1,length(D_train));
        for j=1:length(D_train)
            % evklidska razdalja
            d(j)=sqrt(sum((D_train(j).descriptors-D_test(i).descriptors).^2,'all'));

        end
        [~,idx_min] = min(d);
        if D_train(idx_min).name=="tomato"
            predictions(i) = 0;
        else
            predictions(i) = 1;
        end

        %ground truth:
        if D_test(i).name =="tomato"
            gnd_truth(i) = 0;
        else
            gnd_truth(i) = 1;
        end

    end
end