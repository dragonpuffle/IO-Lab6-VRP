import re


class VRPData:
    def __init__(self, parsed_dict: dict | None = None):
        if parsed_dict is None:
            print('error')
            return

        self.optimal_value = parsed_dict['optimal_value']
        self.num_customers = parsed_dict['num_customers']
        self.num_trucks = parsed_dict['num_trucks']
        self.truck_capacity = parsed_dict['truck_capacity']
        self.coords = parsed_dict['coords']
        self.demands = parsed_dict['demands']
        self.start_id = parsed_dict['start_id']


def read_parse_data(filename: str) -> VRPData:
    with open(filename, 'r') as f:
        lines = f.readlines()

    optimal_value =  None
    num_customers = None
    num_trucks = None
    truck_capacity = None
    coords = {}
    demands = {}
    start_id = None

    for line in lines[:7]:
        if 'Optimal value' in line:
            match = re.search(r'Optimal value\s*:\s*(\d+)', line)
            if match:
                optimal_value = int(match.group(1))
        if 'No of trucks' in line:
            match = re.search(r'No of trucks\s*:\s*(\d+)', line)
            if match:
                num_trucks = int(match.group(1))
        elif line.startswith('DIMENSION'):
            num_customers = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            truck_capacity = int(line.split(':')[1].strip())

    if optimal_value is None or num_trucks is None or num_customers is None or truck_capacity is None:
        print('Some data is missing')
        print('Optimal value:', optimal_value)
        print('No of trucks:', num_trucks)
        print('No of customers:', num_customers)
        print('Truck capacity:', truck_capacity)
        return VRPData()

    for line in lines[7: 7 + num_customers]:
        parts = line.strip().split()
        if len(parts) != 3:
            print('error in parsing coords')
            return VRPData()
        customer_id = int(parts[0]) - 1
        x, y = map(int, parts[1:3])
        coords[customer_id] = (x, y)

    for line in lines[7 + num_customers + 1:7 + num_customers + 1 + num_customers]:
        parts = line.strip().split()
        if len(parts) != 2:
            print('error in parsing demands')
            return VRPData()
        customer_id = int(parts[0]) - 1
        demand = int(parts[1])
        demands[customer_id] = demand
        if demand == 0:
            start_id = customer_id

    if start_id is None:
        print('error: no start id')
        return VRPData()

    if coords is None or demands is None:
        print('error: no coords or demands')
        return VRPData()

    result = {
        'optimal_value':  optimal_value,
        'num_customers': num_customers,
        'num_trucks': num_trucks,
        'truck_capacity': truck_capacity,
        'coords': coords,
        'demands': demands,
        'start_id': start_id,
        }
    return VRPData(result)


if __name__ == '__main__':
    read_parse_data('benchmarks/A/A-n32-k5.vrp')