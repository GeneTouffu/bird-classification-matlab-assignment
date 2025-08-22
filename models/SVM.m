classdef SVM < BaseModel
    methods
        function svm = fit(obj, X_train, y_train, ~, ~)
            svm = fitcecoc(X_train, y_train);
        end

        function y_pred = my_predict(obj, svm, X_test)
            y_pred = predict(svm, X_test);
        end

        function y_pred = evaluate(obj, svm, X_test, y_test)
            y_pred = obj.my_predict(svm, X_test);
        end

        function name = get_name(obj)
            name = 'SVM Classifier';
        end
    end
end
