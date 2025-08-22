classdef (Abstract) BaseModel
    methods (Abstract)
        obj = fit(obj, X_train, y_train, X_val, y_val)

        y_pred = my_predict(obj, X_test)

        acc = evaluate(obj, X_test, y_test)

        name = get_name(obj)
    end
end
