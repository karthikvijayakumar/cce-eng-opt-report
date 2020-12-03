X = [ones(1,91);1:0.1:10]';
true_w = [5;2];
noise_variance = 1
b = X*true_w + (randn(size(X)(1),1)*noise_variance);
num_samples = size(X)(1)

epsilon = 0.1;
alpha = [ 0.01, 0.005, 0.001, 0.0001 ];
beta_param = 0.8;
beta1_param = 0.9;
beta2_param = 0.999;
loss_values_steepest_grad_descent = zeros(length(alpha),91);
loss_values_accelerated_grad_descent = zeros(length(alpha),91);
loss_values_adam = zeros(length(alpha),91);
iter_values = 1:10:1000; % Run gradient descent for 1000 iterations

% Steepest gradient descent
for alpha_counter = 1:length(alpha)
    curr_alpha = alpha(alpha_counter);
    printf("Alpha : %f\n", curr_alpha )
    w = [ 0; 0];
    w_new = [0;0];

    for iter = 1:1000  
        w = w_new;
        loss_function = (1/num_samples)*sum( (X*w-b).^2 ); % Mean squared error
        grad_loss_function = (1/num_samples)*[ 2*(X*w-b)'*X(:,1) ;  2*(X*w-b)'*X(:,2)  ];
        w_new = w - curr_alpha*grad_loss_function;
        iter = iter + 1;
        if ( rem(iter,10) == 0 )
            % printf("Iterations: %d ; Loss: %f \n", iter, loss_function )
            loss_values_steepest_grad_descent(alpha_counter, iter/10) = loss_function;
        endif
    end
    printf("------\n")
end

% Accelerated gradient descent
for alpha_counter = 1:length(alpha)
    curr_alpha = alpha(alpha_counter);
    printf("Alpha : %f\n", curr_alpha )
    w = [ 0; 0];
    w_new = [0;0];
    q = [0;0];
    q_new = [0;0];

    for iter = 1:1000  
        q = q_new;
        w = w_new;
        loss_function = (1/num_samples)*sum( (X*w-b).^2 ); % Mean squared error
        grad_loss_function =(1/num_samples)*[ 2*(X*w-b)'*X(:,1) ;  2*(X*w-b)'*X(:,2)  ];
        q_new = beta_param*q + grad_loss_function;
        w_new = w - curr_alpha*q_new;
        iter = iter + 1;
        if ( rem(iter,10) == 0 )
            % printf("Iterations: %d ; Loss: %f \n", iter, loss_function )
            loss_values_accelerated_grad_descent(alpha_counter, iter/10) = loss_function;
        endif
    end
    printf("------\n")
end

% ADAM
for alpha_counter = 1:length(alpha)
    curr_alpha = alpha(alpha_counter);
    printf("Alpha : %f\n", curr_alpha )
    w = im2double(ones(size(X)(2),1))*0.5;
    m = im2double(zeros(size(X)(2),1));
    v = im2double(zeros(size(X)(2),1));
    w_new = im2double(ones(size(X)(2),1))*0.5;
    m_new = im2double(zeros(size(X)(2),1));
    v_new = im2double(zeros(size(X)(2),1));

    for iter = 1:1000  
        w = w_new;
        m = m_new;
        v = v_new;
    
        loss_function = (1/num_samples)*sum( (X*w-b).^2 ); % Mean squared error
        grad_loss_function = (1/num_samples)*[ 2*(X*w-b)'*X(:,1) ;  2*(X*w-b)'*X(:,2)  ];

        m_new = beta1_param*m + (1-beta1_param)*grad_loss_function;
        v_new = beta2_param*v + (1-beta2_param)*(grad_loss_function.^2);

        m_hat = ( m_new/(1-(beta1_param^iter)) );
        v_hat = ( v_new/(1-(beta2_param^iter)) );
               
        w_new = w - (curr_alpha*(m_hat./(v_hat.^(0.5) + epsilon)));

        if ( rem(iter,10) == 0 )
            printf("Iterations: %d ; Loss: %f \n", iter, loss_function )
            loss_values_adam(alpha_counter, iter/10) = loss_function;
        endif
    end
    printf("------\n")
end

plot_fig = figure();
subplot(3,2,1);
plot(
  iter_values, loss_values_steepest_grad_descent(1,:), "r", "linewidth", 2 ,
  iter_values, loss_values_steepest_grad_descent(2,:), "g", "linewidth", 2 ,
  iter_values, loss_values_steepest_grad_descent(3,:), "b", "linewidth", 2 ,
  iter_values, loss_values_steepest_grad_descent(4,:), "m", "linewidth", 2 
);
ylim([0 50]);
title("Steepest Gradient Descent");
xlabel("Iterations");
ylabel("Loss function");
h = legend( "1e-2", "5e-3", "1e-3", "1e-4" );
set(h, "location" ,"southoutside");
set(h,"orient", "horizontal");
grid("on");

subplot(3,2,2);
semilogy(
  iter_values, loss_values_steepest_grad_descent(1,:), "r", "linewidth", 2 ,
  iter_values, loss_values_steepest_grad_descent(2,:), "g", "linewidth", 2 ,
  iter_values, loss_values_steepest_grad_descent(3,:), "b", "linewidth", 2 ,
  iter_values, loss_values_steepest_grad_descent(4,:), "m", "linewidth", 2 
);
ylim([0 50]);
title("Steepest Gradient Descent");
xlabel("Iterations");
ylabel("Loss function");
h = legend( "1e-2", "5e-3", "1e-3", "1e-4" )
set(h, "location" ,"southoutside");
set(h,"orient", "horizontal");
grid("on");

subplot(3,2,3);
plot(
  iter_values, loss_values_accelerated_grad_descent(1,:), "r", "linewidth", 2 ,
  iter_values, loss_values_accelerated_grad_descent(2,:), "g", "linewidth", 2 ,
  iter_values, loss_values_accelerated_grad_descent(3,:), "b", "linewidth", 2 ,
  iter_values, loss_values_accelerated_grad_descent(4,:), "m", "linewidth", 2 
);
ylim([0 50]);
title("Accelerated Gradient Descent");
xlabel("Iterations");
ylabel("Loss function");
h = legend( "1e-2", "5e-3", "1e-3", "1e-4" )
set(h, "location" ,"southoutside");
set(h,"orient", "horizontal");
grid("on");

subplot(3,2,4);
semilogy(
  iter_values, loss_values_accelerated_grad_descent(1,:), "r", "linewidth", 2 ,
  iter_values, loss_values_accelerated_grad_descent(2,:), "g", "linewidth", 2 ,
  iter_values, loss_values_accelerated_grad_descent(3,:), "b", "linewidth", 2 ,
  iter_values, loss_values_accelerated_grad_descent(4,:), "m", "linewidth", 2 
);
ylim([0 50]);
title("Accelerated Gradient Descent");
xlabel("Iterations");
ylabel("Loss function");
h = legend( "1e-2", "5e-3", "1e-3", "1e-4" );
set(h, "location" ,"southoutside");
set(h,"orient", "horizontal");
grid("on");

subplot(3,2,5);
plot(
  iter_values, loss_values_adam(1,:), "r", "linewidth", 2 ,
  iter_values, loss_values_adam(2,:), "g", "linewidth", 2 ,
  iter_values, loss_values_adam(3,:), "b", "linewidth", 2 ,
  iter_values, loss_values_adam(4,:), "m", "linewidth", 2 
);
ylim([0 50]);
title("ADAM");
xlabel("Iterations");
ylabel("Loss function");
h = legend( "1e-2", "5e-3", "1e-3", "1e-4" );
set(h, "location" ,"southoutside");
set(h,"orient", "horizontal");
grid("on");

subplot(3,2,6);
semilogy(
  iter_values, loss_values_adam(1,:), "r", "linewidth", 2 ,
  iter_values, loss_values_adam(2,:), "g", "linewidth", 2 ,
  iter_values, loss_values_adam(3,:), "b", "linewidth", 2 ,
  iter_values, loss_values_adam(4,:), "m", "linewidth", 2 
);
ylim([0 50]);
title("ADAM");
xlabel("Iterations");
ylabel("Loss function");
h=legend( "1e-2", "5e-3", "1e-3", "1e-4" );
set(h, "location" ,"southoutside");
set(h,"orient", "horizontal");
grid("on");

print(plot_fig, "linear_regression_one_variable.eps", "-depsc");