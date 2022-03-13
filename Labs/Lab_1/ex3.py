import sys




if __name__ == '__main__':
    
    city_numb = {}
    month_numb = {}
    tot_births = 0
    tot_city = 0
    with open(sys.argv[1]) as file:
        for line in file:
            name, surname, city, date = line.split()
            
            tot_births += 1
            if city not in city_numb:
                city_numb[city] = 1
            else:
                city_numb[city] += 1
            
            month = int(line.split('/')[1])
            if month not in month_numb:
                month_numb[month] = 1
            else:
                month_numb[month] += 1
        
    for i in city_numb:
        print('City = %s, #births = %d'  % (i, city_numb[i]))
    
    for i in month_numb:
        print('Month = %s, #births = %d'  % (i, month_numb[i]))
    
    print('Average number of births per city: %f' % (tot_births / len(city_numb)))