function simplified_data = smplfd_dataset()
    gait_ireg_surface_dataset = load('../data.mat').('data');
    
    simplified_data = [];
    
    % iter thru all patients
    ids = fieldnames(gait_ireg_surface_dataset);
    for p=1:numel(ids)
        sensors = gait_ireg_surface_dataset.(ids{p});    % redefine struct for simplicity
        for t=1:57                      % itet thru 57 trials to add each
            if t < 4                    % skip calib trials
                continue
            else
                clear x;
                y = sensors.('trunk').('Surface')(t);
    
                ch = {'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z'};
                seg = {'trunk', 'thighL', 'thighR', 'shankL', 'shankR'};
    
                for s=1:length(seg)
                    x(:,:,s)=cell2mat(table2array(sensors.(seg{s})(t,ch)));
                end
    
                x = permute(x,[1,3,2]);
                entry = {p, x , y};
    
                % split x based on gait events and add each 101x5x6 gait cycle
                % append entry as new cell array row
                simplified_data = [simplified_data; entry];
            end
        end
    end
end

