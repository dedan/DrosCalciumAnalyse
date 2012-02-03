import MySQLdb
from numpy import array
import pylab as plt

'''
BCUTDESC = ['burdenL1', 'burdenL2', 'burdenL3', 'burdenH1', 'burdenH2', 'burdenH3',
            'polariL1', 'polariL2', 'polariL3', 'polariH1', 'polariH2', 'polariH3',
            'chargeL1', 'chargeL2', 'chargeL3', 'chargeH1', 'chargeH2', 'chargeH3',
            'hbondL1', 'hbondL2', 'hbondL3', 'hbondH1', 'hbondH2', 'hbondH3']


IJCDESC = ['LogD', 'TPSA', 'H_bond_acceptors',
            'H_bond_donors', 'avgPol', 'Axxpol', 'Ayypol', 'Azpol', 'molPol',
            'ASAhydrophobic', 'ASAnegative', 'ASAPlus', 'ASAPolar', 'maxProjection',
            'minProjection', 'maxProjectionRad', 'minProjectionRad', 'vdwSurface',
            'HbondaccSide', 'HbonddonSide', 'PiEnergy', 'refractivity', 'min2maxProjectionArea',
             'min2maxProjectionRadius', 'MAXarea2radius', 'MINarea2radius', 'VP_ACD_mmHG25',
             'minAtomCharge', 'maxAtomCharge', 'oxygens'] 

#Strongest_acidic_pKa and Strongest_basic_pKa have none values

IJCDESCnew = ['mass', 'relAliphaticCount', 'relAliphaticRingCount', 'relAromaticRingCount', 'relAromaticCount', 'relRingCount',
              'relAsymmetricCount', 'relRotableBond', 'topPolarSfArea', 'ASA', 'relASApolar', 'relASAnegative', 'relAsaplus', 'hmoPiEnergy', 'logP']

IJCgraph = ['balabanIndex', 'hararyIndex', 'hyperWienerIndex', 'wienerIndex', 'plattIndex', 'randicIndex']

BCUTDESC = ['BCUT_burdenL1', 'BCUT_burdenL2', 'BCUT_burdenL3', 'BCUT_burdenH1', 'BCUT_burdenH2', 'BCUT_burdenH3',
            'BCUT_polariL1', 'BCUT_polariL2', 'BCUT_polariL3', 'BCUT_polariH1', 'BCUT_polariH2', 'BCUT_polariH3',
            'BCUT_chargeL1', 'BCUT_chargeL2', 'BCUT_chargeL3', 'BCUT_chargeH1', 'BCUT_chargeH2', 'BCUT_chargeH3',
            'BCUT_hbondL1', 'BCUT_hbondL2', 'BCUT_hbondL3', 'BCUT_hbondH1', 'BCUT_hbondH2', 'BCUT_hbondH3']

DRAGONDESC = ['GGI1', 'GGI2', 'GGI3', 'GGI4', 'GGI5', 'GGI6', 'GGI7', 'GGI8', 'GGI9', 'GGI10',
              'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8', 'JGI9', 'JGI10', 'JGT', 'W3D',
              'J3D', 'H3D', 'AGDD', 'DDI', 'ADDD', 'G1', 'G2', 'RGyr', 'SPAN', 'SPAM', 'MEcc', 'SPH', 'ASP',
              'FDI', 'PJI3', 'L_Bw', 'SEig', 'HOMA', 'RCI', 'AROM', 'HOMT', 'DISPm', 'QXXm', 'QYYm', 'QZZm',
              'DISPv', 'QXXv', 'QYYv', 'QZZv', 'DISPe', 'QXXe', 'QYYe', 'QZZe', 'DISPp', 'QXXp', 'QYYp',
              'QZZp', 'G_OO', 'G_OS'] #, 'G_SS']
'''

IJCDESC = ['avgPol', 'maxPolTens', 'minPolTens', 'molPol', 'numOxy', 'maxSigmaEneg', 'minSigmaEneg',
 'maxPiEneg', 'minPiEneg', 'maxAtomCharge', 'minAtomCharge', 'aliphaticAtomCount', 'aliphaticRingCount',
 'aliphaticBondCount', 'aromaticAtomCount', 'aromaticBondCount', 'aromaticRingCount', 'ASAHydrophobic',
 'ASANegative', 'ASAPlus', 'ASAPolar', 'balabanIndex', 'carboaliphaticRingCount', 'carboaromaticRingCount',
 'carboRingCount', 'chainAtomCount', 'chainBondCount', 'cyclomaticNumber', 'dreidingEnergy', 'fusedAliphaticRingCount',
 'fusedAromaticRingCount', 'fusedRingCount', 'heteroaliphaticRingCount', 'heteroaromaticRingCount',
 'hetroRingCount', 'hyperWienerIndex', 'maximalProjectionArea', 'minimalProjectionArea', 'maximalProjectionRadius',
 'minimalProjectionRadius', 'plattIndex', 'randicIndex', 'ringAtomCount', 'largestRingSystemSize', 'RINGATOMCOUNT_2',
 'ringBondCount', 'szedgedIndex', 'stereoDoubleBondCount', 'PSA', 'vanDerWaalsSurfaceArea', 'ASA', 'wienerIndex',
 'wienerPolarity', 'acceptorCount', 'acceptorSiteCount', 'donorCount', 'donorSiteCount', 'hmoPiEnergy',
 'piEnergy', 'logP', 'LogD', 'refractivity']




class instantJChemInterface(object):
    def __init__(self, where='localhost', who='jan', db='ODORDB'):
        self.conn = MySQLdb.connect(where, who, '', db)

    def get_descriptors(self, table):
        cursor = self.conn.cursor()
        cursor.execute("SHOW COLUMNS FROM " + table)
        out = cursor.fetchall()
        out = [i[0] for i in out][3:]
        cursor.close()
        return out

    def fetch_data(self, rows, table):
        ''' returns the data of rows in table'''
        rowstr = ('%s, ' * len(rows) % tuple(rows))[:-2] 
        cursor = self.conn.cursor()
        cursor.execute("SELECT " + rowstr + " FROM " + table)
        out = cursor.fetchall()
        cursor.close()
        return out
    
    def fetch_selected_data(self, table, cols_out, col_select, val_select):
        ''' returns the columns which match val_select'''
        colstr = ('%s, ' * len(cols_out) % tuple(cols_out))[:-2]
        select = []
        seperator = ' AND '
        for ind, col in enumerate(col_select):
            val = val_select[ind] if len(col_select) == len(val_select) else val_select[0]
            select.append(col + '%s' % val)
        select = seperator.join(select)
        cursor = self.conn.cursor() 
        #print 'SELECT %s FROM %s WHERE %s' % (colstr, table, select)
        cursor.execute('SELECT %s FROM %s WHERE %s' 
                                % (colstr, table, select))
        out = cursor.fetchall()
        cursor.close()
        return out
    
    def make_table_dict(self, id, desc, table, col_select=['True'], val_select=['']):
        ''' id (= dic key), property, table, [col_select], [val_select]'''     
        tabledict = {}
        raw_data = self.fetch_selected_data(table, [id] + desc, col_select, val_select)
        for sample in raw_data:
            try:
                tabledict[sample[0]].append(list(sample[1:]))
            except KeyError:
                tabledict[sample[0]] = [list(sample[1:])]    
        return tabledict

    def write_data(self, table_name, col_name, col_type, idcol, id_dict):
        cursor = self.conn.cursor()
        #cursor.execute('ALTER TABLE %s ADD %s %s' % (table_name, col_name, col_type))
        for (id, data) in id_dict.items():
            cursor.execute('SELECT %s FROM %s WHERE %s=%s' % (idcol, table_name, idcol, id))
            if cursor.fetchall():
                sqlcommand = 'UPDATE %s SET %s="%' + '%s"' % col_type + ' WHERE %s="%s"'
                #print data, sqlcommand % (table_name, col_name, data, idcol, id)
                cursor.execute(sqlcommand % (table_name, col_name, data, idcol, id))
            else:
                #print 'INSERT INTO %s (%s, %s) VALUES (%s, "%s")' % (table_name, idcol, col_name, id, data)
                cursor.execute('INSERT INTO %s (%s, %s) VALUES (%s, "%s")' % (table_name, idcol, col_name, id, data)) 
        cursor.close()
        self.conn.commit()      

    def filter_dict(self, response_dict, feature_dict):
        for key in response_dict.keys():
            if response_dict[key] == ['?']:
                response_dict.pop(key)
                feature_dict.pop(key) 
        

class DataTracking(object):
    
    def __init__(self):
        self.pos_to_id_map = {}
        self.id_to_pos = {}
    
    def dataset_generation(self, feature_dict, response_dict):
        targets = []
        features = []
        for key in response_dict.keys():
            for response in response_dict[key]:
                features.append(feature_dict[key])
                if response[0] <> '?':
                    targets.append(2 * int(response[0]) - 1)
                else:
                    targets.extend([-1, 1])
                    features.append(feature_dict[key])
                    self.pos_to_id_map[len(targets) - 2] = key
                self.pos_to_id_map[len(targets) - 1] = key
                print array(features).squeeze().transpose().shape, array(targets).shape
        return (array(features).squeeze().transpose(), array(targets))
    
    def featureset_generation(self, feature_dict):
        features = []
        for (key, value) in feature_dict.items():
            features.append(value)
            self.pos_to_id_map[len(features) - 1] = key
            self.id_to_pos[key] = len(features) - 1
        return array(features).squeeze().transpose()
        
    def convert_datamap_to_idmap(self, pos_to_data_map):
        dict_out = {}
        for (poskey, data) in pos_to_data_map.items():  
                try:
                    assert(dict_out[self.pos_to_id_map[poskey]] == data)
                except KeyError:
                    dict_out[self.pos_to_id_map[poskey]] = data
        return dict_out
    
    def convert_datalist_to_idmap(self, datalist, multi=False):
        datalist = zip(*datalist) if multi else datalist
        dict_out = {}
        for list_pos, data in enumerate(datalist):
            dict_out[self.pos_to_id_map[list_pos]] = data
        return dict_out
    
    def aggregate(self, idmap, aggregator):
        aggregated_dict = {}
        for (id, data) in idmap.items():
            aggregated_dict[id] = aggregator(data)
        return aggregated_dict
    
    def hist_idmap(self, idmap, path, bins=10, range=[0, 1]):
        
        plt.rcParams.update({'axes.labelsize': 20,
                             'xtick.labelsize': 20,
                             'ytick.labelsize': 20,
                            })
        for (id, data) in idmap.items():
            plt.clf()
            plt.hist(data, bins, range, normed=True, histtype='bar', align='mid')
            plt.savefig(path + str(id) + '.png')
    
    def identity_fct(self, object):
        return object
    
    def multi_to_single_dict(self, multi_dict):
        elements = len(multi_dict.values()[0])
        for count in range(elements):
            single_dict = {}
            for (key, value) in multi_dict.items():
                single_dict[key] = value[count]
            yield single_dict
