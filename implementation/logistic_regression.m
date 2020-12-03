sigmoid = @(x) (1+exp(-x)).^(-1);

range = linspace(0,10,101);
[grid_x, grid_y, grid_z] = meshgrid(range, range, range);

num_samples = 101*101*101;
trainX = [ reshape(grid_x,1,num_samples); reshape(grid_y,1,num_samples); reshape(grid_z,1,num_samples); ones(1, num_samples)]';

true_w = [1.0;-1.0;0.0;1.0];
% Modelling the plane  x-y+1 >= 0

noise_variance = 0;
trainY = (trainX*true_w) >= 0;
% trainY = (trainX*true_w + (randn(num_samples,1)*noise_variance)) >= 0;

num_positive_samples_train = sum(trainY == 1);
printf("Num positive training samples: %d \n", num_positive_samples_train)
printf("Prop of positive samples: %f \n", (num_positive_samples_train/num_samples) )

epsilon = 0.00000001;
alpha = [ 0.001,  0.01, 0.1 ];
beta_param = 0.5;
beta1_param = 0.9;
beta2_param = 0.999;

lambda = 0.5;
iter_values = 1:100;
loss_values_steepest_gd = zeros(length(alpha),length(iter_values));
loss_values_accelerated_gd = zeros(length(alpha),length(iter_values));
loss_values_adam = zeros(length(alpha),length(iter_values));

% Steepest gradient descent
printf("Running steepest gradient descent \n")
for alpha_counter = 1:3
    curr_alpha = alpha(alpha_counter);
    printf("Alpha : %f\n", curr_alpha )
    w = im2double(ones(size(trainX)(2),1))*0.5;
    w_new = im2double(ones(size(trainX)(2),1))*0.5;

    loss_function = (-1/num_samples)*( (trainY'*log( sigmoid(trainX*w) )) + ((1-trainY)'*log( 1-sigmoid(trainX*w) )) ) + lambda*(w'*w);
    printf("Iterations: 0 ; Loss: %f \n", loss_function )

    for iter = 1:100
        w = w_new;
        loss_function = (-1/num_samples)*( (trainY'*log( sigmoid(trainX*w) )) + ((1-trainY)'*log( 1-sigmoid(trainX*w) )) ) + lambda*(w'*w);
        grad_loss_function = (-1/num_samples)*( trainX'*(trainY - sigmoid(trainX*w)) ) + 2*lambda*w;

        w_new = w - curr_alpha*grad_loss_function;
        loss_values_steepest_gd(alpha_counter, iter) = loss_function;

        if ( rem(iter,10) == 0 )
            num_positive_predictions = sum( sigmoid(trainX*w)>= 0.5 );
            trainY_pred = sigmoid(trainX*w)>= 0.5;
            accuracy = sum(trainY_pred == trainY)/num_samples;
            precision = sum(trainY(trainY_pred==1))/sum(trainY_pred==1);
            recall = sum(trainY_pred(trainY==1))/sum(trainY==1);
            printf("Iterations: %d ; Loss: %f; #+ve preds: %d; Prec: %f; Recall: %f; Accuracy: %f \n", iter, loss_function, num_positive_predictions, precision, recall, accuracy )
      endif
    end
    printf("------\n")
end

% Accelerated gradient descent
printf("Running accelerated gradient descent \n")
for alpha_counter = 1:3
    curr_alpha = alpha(alpha_counter);
    printf("Alpha : %f\n", curr_alpha )
    w = im2double(ones(size(trainX)(2),1))*0.5;
    q = im2double(ones(size(trainX)(2),1))*0.5;
    w_new = im2double(ones(size(trainX)(2),1))*0.5;
    q_new = im2double(ones(size(trainX)(2),1))*0.5;

    loss_function = (-1/num_samples)*( (trainY'*log( sigmoid(trainX*w) )) + ((1-trainY)'*log( 1-sigmoid(trainX*w) )) ) + lambda*(w'*w);
    printf("Iterations: 0 ; Loss: %f \n", loss_function )

    for iter = 1:100
        w = w_new;
        q = q_new;
        loss_function = (-1/num_samples)*( (trainY'*log( sigmoid(trainX*w) )) + ((1-trainY)'*log( 1-sigmoid(trainX*w) )) ) + lambda*(w'*w);
        grad_loss_function = (-1/num_samples)*( trainX'*(trainY - sigmoid(trainX*w)) ) + 2*lambda*w;

        q_new = beta_param*q + grad_loss_function;
        w_new = w - curr_alpha*q_new;
        loss_values_accelerated_gd(alpha_counter, iter) = loss_function;

        if ( rem(iter,10) == 0 )
            num_positive_predictions = sum( sigmoid(trainX*w)>= 0.5 );
            trainY_pred = sigmoid(trainX*w)>= 0.5;
            accuracy = sum(trainY_pred == trainY)/num_samples;
            precision = sum(trainY(trainY_pred==1))/sum(trainY_pred==1);
            recall = sum(trainY_pred(trainY==1))/sum(trainY==1);
            printf("Iterations: %d ; Loss: %f; #+ve preds: %d; Prec: %f; Recall: %f; Accuracy: %f \n", iter, loss_function, num_positive_predictions, precision, recall, accuracy )
      endif
    end
    printf("------\n")
end

% ADAM
printf("Running ADAM \n")
for alpha_counter = 1:3
    curr_alpha = alpha(alpha_counter);
    printf("Alpha : %f\n", curr_alpha )
    w = im2double(ones(size(trainX)(2),1))*0.5;
    m = im2double(zeros(size(trainX)(2),1));
    v = im2double(zeros(size(trainX)(2),1));
    w_new = im2double(ones(size(trainX)(2),1))*0.5;
    m_new = im2double(zeros(size(trainX)(2),1));
    v_new = im2double(zeros(size(trainX)(2),1));

    loss_function = (-1/num_samples)*( (trainY'*log( sigmoid(trainX*w) )) + ((1-trainY)'*log( 1-sigmoid(trainX*w) )) ) + lambda*(w'*w);
    printf("Iterations: 0 ; Loss: %f \n", loss_function )

    for iter = 1:100
        w = w_new;
        m = m_new;
        v = v_new;

        loss_function = (-1/num_samples)*( (trainY'*log( sigmoid(trainX*w) )) + ((1-trainY)'*log( 1-sigmoid(trainX*w) )) ) + lambda*(w'*w);
        grad_loss_function = (-1/num_samples)*( trainX'*(trainY - sigmoid(trainX*w)) ) + 2*lambda*w;

        m_new = beta1_param*m + (1-beta1_param)*grad_loss_function;
        v_new = beta2_param*v + (1-beta2_param)*(grad_loss_function.^2);

        m_hat = ( m_new/(1-(beta1_param^iter)) );
        v_hat = ( v_new/(1-(beta2_param^iter)) );
               
        w_new = w - (curr_alpha*(m_hat./(v_hat.^(0.5) + epsilon)));

        % printf("m_new : %s \n", disp(m_new))
        % printf("v_new : %s \n", disp(v_new))
        % printf("w_new : %s \n", disp(w_new))
        loss_values_adam(alpha_counter, iter) = loss_function;
        if ( rem(iter,10) == 0 )
            num_positive_predictions = sum( sigmoid(trainX*w)>= 0.5 );
            trainY_pred = sigmoid(trainX*w)>= 0.5;
            accuracy = sum(trainY_pred == trainY)/num_samples;
            precision = sum(trainY(trainY_pred==1))/sum(trainY_pred==1);
            recall = sum(trainY_pred(trainY==1))/sum(trainY==1);
            printf("Iterations: %d ; Loss: %f; #+ve preds: %d; Prec: %f; Recall: %f; Accuracy: %f \n", iter, loss_function, num_positive_predictions, precision, recall, accuracy )
      endif
    end
    printf("------\n")
end

plot_fig = figure()
subplot(3,1,1)
semilogy(
  iter_values, loss_values_steepest_gd(1,:), "r", "linewidth", 2 ,
  iter_values, loss_values_steepest_gd(2,:), "g", "linewidth", 2 ,
  iter_values, loss_values_steepest_gd(3,:), "b", "linewidth", 2
)
title("Loss function vs iteration - Steepest Gradient Descent")
ylim([0.3 5])
grid("on")
xlabel("Iterations")
ylabel("Loss function")
legend("0.001", "0.01", "0.1")

subplot(3,1,2)
semilogy(
  iter_values, loss_values_accelerated_gd(1,:), "r", "linewidth", 2 ,
  iter_values, loss_values_accelerated_gd(2,:), "g", "linewidth", 2 ,
  iter_values, loss_values_accelerated_gd(3,:), "b", "linewidth", 2 
)
title("Loss function vs iteration - Accelerated Gradient Descent")
ylim([0.3 5])
grid("on")
xlabel("Iterations")
ylabel("Loss function")
legend("0.001", "0.01", "0.1")

subplot(3,1,3)
semilogy(
  iter_values, loss_values_adam(1,:), "r", "linewidth", 2 ,
  iter_values, loss_values_adam(2,:), "g", "linewidth", 2 ,
  iter_values, loss_values_adam(3,:), "b", "linewidth", 2 
)
title("Loss function vs iteration - ADAM")
xlabel("Iterations")
ylabel("Loss function")
ylim([0.3 5])
grid("on")
legend("0.001", "0.01", "0.1")

print(plot_fig, "logistic_regression.eps", "-depsc")