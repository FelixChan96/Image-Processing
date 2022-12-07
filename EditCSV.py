from csv import writer
List = ['G7',1240,17]
with open('machinesdata.csv','a') as f_object:
    writer_object=writer(f_object)
    writer_object.writerow(List)
    f_object.close()      
    
    