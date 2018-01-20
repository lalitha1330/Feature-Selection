filename='/home/lalitha/testdata3.txt';
data = importdata(filename,' ');
data = data(randperm(size(data, 1)), :);
current_set_of_features = []; % Initialize an empty set
accuracy_list=[];
 
for i = 1 : size(data,2)-1 
    disp(['On the ',num2str(i),'th level of the search tree'])
    feature_to_add_at_this_level = [];
    best_so_far_accuracy = 0;    
    
     for k = 1 : size(data,2)-1 
       if isempty(intersect(current_set_of_features,k)) % Only consider adding, if not already added.
        disp(['--Considering adding the ', num2str(k),' feature'])
        accuracy = leave_one_out_cross_validation(data,current_set_of_features,k);
        
        if accuracy > best_so_far_accuracy 
            best_so_far_accuracy = accuracy;
            feature_to_add_at_this_level = k;            
        end        
      end
     end
    accuracy_list=[accuracy_list best_so_far_accuracy];
    current_set_of_features(i) =  feature_to_add_at_this_level;
    disp(['On level ', num2str(i),' i added feature ', num2str(feature_to_add_at_this_level), ' to current set'])
    disp(['the accuracy at level ', num2str(i),' is ', num2str(best_so_far_accuracy)])
end
 for i=1:size(current_set_of_features,2)
           disp(['the features added are ', num2str(current_set_of_features(i))]);
 end
 disp(['the maximum accuracy is ', num2str(max(accuracy_list))]);

function accuracy = leave_one_out_cross_validation(data,current_set_of_features,k)
correct=0;
data_TRAIN = data;
data_TEST = data;
TRAIN=[];
TEST=[];

TRAIN_class_labels=data_TRAIN(:,1);
TEST_class_labels=data_TEST(:,1);
    for i= 1 :size(current_set_of_features,2)
      TRAIN = [TRAIN data_TRAIN(:,current_set_of_features(i)+1)];
    end
TRAIN=[TRAIN data_TRAIN(:,k+1)];
    for i= 1 : size(current_set_of_features,2)
      TEST = [TEST data_TEST(:,current_set_of_features(i)+1)];
    end
TEST=[TEST data_TEST(:,k+1)];
for i=1:length(TEST_class_labels)
    classify_this_object=TEST(i,:);
    j=i;
    this_objects_actual_class=TEST_class_labels(i);
    predicted_class=knnclassify(TRAIN,TRAIN_class_labels,classify_this_object,j);
 if predicted_class==this_objects_actual_class
     correct=correct+1;
 end
end
accuracy=correct/length(TEST_class_labels);
end

function predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels,unknown_object,j)
best_so_far = inf;
for i = 1 : length(TRAIN_class_labels)
   if(i~=j)
    compare_to_this_object = TRAIN(i,:);
distance = sqrt(sum((compare_to_this_object - unknown_object).^2));
    if distance < best_so_far
      predicted_class = TRAIN_class_labels(i);
     best_so_far = distance;
    end
   end
end
end
