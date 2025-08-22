classdef KNN < BaseModel
    methods
        function knn = fit(obj, X_train, y_train, ~, ~)
            % Train a k-Nearest Neighbors classifier
            knn = fitcknn(X_train, y_train, 'NumNeighbors', 8);
        end

        function y_pred = my_predict(obj, knn, X_test)
            % Predict labels using the trained KNN model
            y_pred = predict(knn, X_test);
        end

        function y_pred = evaluate(obj, knn, X_test, y_test)
            % Evaluate prediction accuracy
            y_pred = obj.my_predict(knn, X_test);
        end

        function name = get_name(obj)
            name = 'KNN Classifier';
        end
    end
end
