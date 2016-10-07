function Solver = modelconfig(model_path, FILTER_TYPE, mode)
if strcmp(mode,'train')
    solver_file = fullfile(model_path,'LRNN_solver.prototxt');
else
    solver_file = fullfile(model_path,'LRNN_solver_test.prototxt');
end
save_file = fullfile(model_path,sprintf('LRNN_solver4%s.mat',FILTER_TYPE));
if ~exist(save_file, 'file')
    Solver = SolverParser(solver_file);
else
    Solver = SolverParser(solver_file, save_file);
end
if caffe('is_initialized') == 0
    if strcmp(mode,'train')
        caffe('init', solver_file, 'train');
        if isfield(Solver, 'model')
            layers = Solver.model;
            caffe('set_weights', layers,'train');
            caffe('set_iter', Solver.iter);
        end
    else
        if ~isfield(Solver, 'model')
            error('you need a trained solver.');
        else
            caffe('init',Solver.net, 'test');
            layers = Solver.model;
            caffe('set_weights', layers, 'test');
        end
    end
end
fprintf('Done with init\n');
% GPU
if strcmp(Solver.solver_mode,'GPU')
    fprintf('Using GPU Mode\n');
    caffe('set_mode_gpu');
    caffe('set_device',Solver.device_id);
else
    fprintf('Using CPU Mode\n');
    caffe('set_mode_cpu');
end
caffe('get_device');
fprintf('Done with set mode\n');

end