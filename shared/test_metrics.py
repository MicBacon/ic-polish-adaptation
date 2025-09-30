from MetricComputer import MetricComputer

mc = MetricComputer()
preds = ["Czerwony samochód jedzie po ulicy."]
refs  = [["Na ulicy jedzie czerwone auto.", "Czerwone auto na drodze."]]
print(mc.compute_metrics_fast(preds, refs))