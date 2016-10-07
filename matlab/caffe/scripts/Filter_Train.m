% Sifei Liu, 10/04/2016
% sliu32@ucmerced.edu
% Learn any type of image filters.
% FILTER_TYPE: supports the following methods:
%               'L0', 'shock', 'wls', 'WMF', 'RTV', 'RGF'
% TRAINI_PATH: path of the training folder, specified by users
% Solver:      solver configures and model parameters
% this version supports LRNN_v1.prototxt; in order to apply 
% LRNN_v2.prototxt u need to change "Gen_training_data_v1" in line 26 to 
% "Gen_training_data_v2"
function Solver = Filter_Train(FILTER_TYPE, TRAINI_PATH)
model_path = '../models';
Solver = modelconfig(model_path, FILTER_TYPE, 'train');
Solver = dataconfig(Solver, TRAINI_PATH);
log_file = fullfile(model_path,sprintf('log_%s.txt', FILTER_TYPE));
if exist(log_file,'file') && ~isfield(Solver,'iter')
    delete(log_file);
end
if isfield(Solver,'model')
    begin = Solver.iter+1;
else
    begin = 1;
end

for iter = begin: Solver.max_iter
    Solver.iter = iter;
    [batch, gt] = Gen_training_data_v1( Solver, FILTER_TYPE );
    batchc = {single(batch)};
    active = caffe('forward',batchc);
    delta = cell(size(active));
    for c = 1:length(active)
        active_ = active{c};
        delta{c} = zeros(size(active{c}));
        [delta_, loss] = L2Loss_hardsample(active_, gt, 'train', 1);
        if size(active_,1)==Solver.patchsize
            Solver.loss(iter) = loss(1);
            Solver.lossg(iter) = loss(2);
        end
        delta{c} = delta_;
    end
    if ~mod(iter,10)
        fin = fopen(log_file,'a+');
        fprintf('========Processed iter %.6d, ',iter);
        fprintf(fin,'========Processed iter %.6d, ',iter);
        if isfield(Solver, 'loss')
            fprintf('loss: %d=======', mean(Solver.loss(iter-9:iter)));
            fprintf(fin,'loss: %d=======', mean(Solver.loss(iter-9:iter)));
        end
        if isfield(Solver, 'lossg')
            fprintf('loss_grad: %d=======', mean(Solver.lossg(iter-9:iter)));
            fprintf(fin,'loss_grad: %d=======', mean(Solver.lossg(iter-9:iter)));
        end
        fprintf('\n');fprintf(fin,'\n');
        fclose(fin);
        [out,psnrs] = showresults(active_, batch(:,:,1:3,:), gt);
        mean_psnr = mean(psnrs);
        % option: save('out.mat','out');
        fprintf('mean_psrn: %.2d\n', mean_psnr);
        layers = caffe('get_weights','train');
        Solver.model = layers;
        save(save_file,'Solver');
    end
    if ~isnan(Solver.loss(iter))
        caffe('backward', delta);
        caffe('update');
    else
        error('Model NAN.')
    end
    if iter == 1
        data = caffe('get_all_data');
        T1_checkdata(data);
        clear data
    end
end