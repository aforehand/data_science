import pandas as pd 
import selenium
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import Select
import re
from collections import defaultdict

class PlantRecommender:
    """A recommender for designing plant guilds and permaculture gardens.
    Uses a REST API for the USDA Plants Database 
    (https://github.com/sckott/usdaplantsapi/) and the National Gerdening 
    Association's plant database (https://garden.org) to recommend plants for 
    various uses.

    Parameters
    ----------
    soil_texture: string, default='medium'
        Values can be {'coarse', 'medium', 'fine'}, or from the Soil Texture 
        Triangle, {'sand', 'coarse_sand', 'fine_sand', 'loamy_coarse_sand', 
        'loamy_fine_sand', 'loamy_very_fine_sand', 'very_fine_sand', 
        'loamy_sand', 'silt', 'sandy_clay_loam', 'very_fine_sandy_loam', 
        'silty_clay_loam', 'silt_loam', 'loam', 'fine_sandy_loam', 'sandy_loam', 
        'coarse_sandy_loam', 'clay_loam', 'sandy_clay', 'silty_clay', 'clay'}.
    
    ph: float, default=6.5
        Soil pH can range from 3.5-9.0. Most plants thrive in soil between 6-7.

    moisture: {'high', 'medium' 'low'}, default='medium'
        How much moisture is available.

    zone: int, default=7
        USDA hardiness zone. Mapped to min_temp because USDA database uses 
        minimum temperature, but hardiness zone is more commonly used.

    region: {'northeast', 'southeast', 'midwest', 'plains', 'pacific'}, default=None
        Set to find plants that are native to a specific region.

    state: two-letter state sbbreviations, default=None
        set to find plants that are native to a specific state.
    """

    def __init__(self, soil_texture='medium', ph=6.5, moisture='medium',  
                zone=7, region=None, state=None):      
        # site parameters      
        if soil_texture in {'coarse', 'sand', 'coarse_sand', 'fine_sand', 
                            'loamy_coarse_sand', 'loamy_fine_sand', 
                            'loamy_very_fine_sand', 'very_fine_sand', 
                            'loamy_sand'}:
            self.soil_texture = {'Adapted_to_Coarse_Textured_Soils': 'Yes'}
        elif soil_texture in {'medium', 'silt', 'sandy_clay_loam', 'very_fine_sandy_loam', 'silty_clay_loam', 
                            'silt_loam', 'loam', 'fine_sandy_loam', 'sandy_loam', 
                            'coarse_sandy_loam', 'clay_loam'}:
            self.soil_texture = {'Adapted_to_Medium_Textured_Soils': 'Yes'}
        elif soil_texture in {'fine', 'sandy_clay', 'silty_clay', 'clay'}:
            self.soil_texture = {'Adapted_to_Fine_Textured_Soils': 'Yes'}
        else:
            self.soil_texture = None
        self.ph = ph
        self.moisture = moisture
        hardiness_zone_to_temp = {1:-60, 2:-50, 3:-40, 4:-30, 5:-20, 6:-10, 7:0,
                                8:10, 9:20, 10:30}
        if zone is not None:
            self.min_temp = hardiness_zone_to_temp[zone]
        else:
            self.min_temp = None
        if region is not None:
            regions = {'northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 
                                'PA', 'DE', 'MD', 'WV', 'VA'], 
                        'southeast': ['NC', 'TN', 'AR', 'SC', 'GA', 'AL', 'MS', 
                                'LA', 'FL'], 
                        'midwest': ['MN', 'WI', 'MI', 'IA', 'IL', 'IN', 'OH', 'MO', 
                                'KY'],
                        'plains': ['MT', 'ND', 'WY', 'SD', 'NE', 'CO', 'KS', 'NM', 
                                'TX', 'OK'],
                        'pacific': ['WA', 'OR', 'ID', 'CA', 'NV', 'UT', 'AZ']}
            self.region = regions[region]
        else:
            self.region = region
        self.state = state 
        self.categorical_attributes = ['Genus', 'Species', 'Varieties']
        self.boolean_attributes = ['Coarse Soil', 'Medium Soil', 'Fine Soil']
        self.numeric_attributes = []

        options = Options()
        options.headless = True
        self.driver = Firefox(options=options)
        self.usda_url = 'https://plantsdb.xyz/search'
        self.garden_search_url = 'https://garden.org/plants/search/advanced.php'
        self.driver.get(self.garden_search_url)
        # garden.org search parameters
        self.sections = [s for s in self.driver.find_elements_by_xpath('//p') if
            s.text is not '']
        self.plant_habit = self.get_inputs(self.sections[0])
        self.life_cycle = Select(self.sections[1].find_element_by_xpath('.//select'))
        self.categorical_attributes.append('Life cycle')
        self.light = self.get_inputs(self.sections[2])
        self.water = self.get_inputs(self.sections[3])
        self.soil_ph = self.get_inputs(self.sections[4])
        self.cold_hardiness = Select(self.sections[5].find_element_by_xpath('.//select'))
        self.numeric_attributes.append('Minimum cold hardiness')
        self.maximum_zone = Select(self.sections[6].find_element_by_xpath('.//select'))
        self.numeric_attributes.append('Maximum recommended zone')
        self.plant_height = self.sections[7].find_element_by_xpath('.//input')
        self.numeric_attributes.append('Plant Height')
        self.plant_spread = self.sections[8].find_element_by_xpath('.//input')
        self.numeric_attributes.append('Plant Spread')
        self.leaves = self.get_inputs(self.sections[9], 'Leaves_')
        self.fruit = self.get_inputs(self.sections[10], 'Fruit_')
        self.fruiting_time = self.get_inputs(self.sections[11], 'Fruiting Time_')
        self.flowers = self.get_inputs(self.sections[12], 'Flowers_')
        self.flower_color = self.get_inputs(self.sections[13],)
        self.bloom_size = self.get_inputs(self.sections[14])
        self.flower_time = self.get_inputs(self.sections[15], 'Flower Time_')
        self.inflorescence_height = self.sections[16].find_element_by_xpath('.//input')
        self.numeric_attributes.append('Inflorescence Height')
        self.foliage_mound_height = self.sections[17].find_element_by_xpath('.//input')
        self.numeric_attributes.append('Foliage Mound Height')
        self.roots = self.get_inputs(self.sections[18])
        self.locations = self.get_inputs(self.sections[19])
        self.uses = self.get_inputs(self.sections[20])
        self.edible_parts = self.get_inputs(self.sections[21])
        self.eating_methods = self.get_inputs(self.sections[22])
        self.dynamic_accumulator = self.get_inputs(self.sections[23])
        self.wildlife_attract = self.get_inputs(self.sections[24])
        self.resistances = self.get_inputs(self.sections[25])
        self.toxicity = self.get_inputs(self.sections[26])
        # some of these have additional inputs once selected.
        # not implemented yet.
        self.propagation_seed = self.get_inputs(self.sections[27])
        self.propagation_other = self.get_inputs(self.sections[28])
        self.pollinators = self.get_inputs(self.sections[29])
        self.containers = self.get_inputs(self.sections[30])
        self.misc = self.get_inputs(self.sections[31])
        self.awards = self.get_inputs(self.sections[32])
        self.conservation_status = Select(self.sections[33].find_element_by_xpath('.//select'))
        self.parentage = self.sections[34].find_element_by_xpath('.//input')
        self.child_plants = self.sections[35].find_element_by_xpath('.//input')
        self.sort_by = Select(self.sections[36].find_element_by_xpath('.//select'))
        self.clear_form = self.sections[37].find_element_by_xpath('.//a')

    def get_inputs(self, section, field=''):
        inputs = section.find_elements_by_xpath('.//input')
        labels = section.find_elements_by_xpath('.//label')
        self.boolean_attributes += [field+l.text for l in labels]
        return {l.text: i for l,i in zip(labels,inputs)}

    def get_results(self):
        links = self.driver.find_elements_by_xpath('.//a')
        results = defaultdict(list)
        for l in links:
            url = l.get_attribute('href')
            name = re.findall(r'(?<=\()([A-Z]\w+ [a-z]\w+)', l.text)
            if 'plants/view/' in url and len(name) > 0:
                results[name[0]].append(url)
        return results

    
    def filter_plants(self, results):
        plants = pd.DataFrame()
        for name in results.keys():
            plant = {a:None for a in (self.categorical_attributes
                                    + self.boolean_attributes
                                    + self.numeric_attributes)}
            genus,species = name.split()
            self.driver.get(f'{self.usda_url}?Genus={genus}&Species={species}')
            self.driver.implicitly_wait(5)
            self.driver.find_element_by_id('rawdata-tab').click()
            data = self.driver.find_element_by_class_name('data')
            try:
                data = eval(data.text.replace('null', 'None'))['data'][0]
                is_native = ('L48 (N)' in data['Native_Status'])
                states = data['State_and_Province']
                states = states[states.index('(')+1:states.index(')')] 
                in_location = ((self.state is None and self.region is None) or 
                                (self.state in states) or 
                                (len(set(self.region) & set(states)) > 0))
                in_zone = self.min_temp >= eval(data['Temperature_Minimum_F'])
                in_ph_range = ((self.ph >= eval(data['pH_Minimum'])) and 
                                (self.ph <= eval(data['pH_Maximum'])))
            except:
                continue
            if is_native and in_location and in_zone and in_ph_range:
                plant['Genus'] = genus
                plant['Species'] = species
                plant['Varieties'] = results[name]
                plant['Coarse Soil'] = data['Adapted_to_Coarse_Textured_Soils']=='Yes'
                plant['Medium Soil'] = data['Adapted_to_Medium_Textured_Soils']=='Yes'
                plant['Fine Soil'] = data['Adapted_to_Fine_Textured_Soils']=='Yes'            
                self.driver.get(results[name][0])
                table = self.driver.find_element_by_xpath('//caption[contains(text(),'
                        ' "General Plant Information")]/../tbody')
                rows = table.find_elements_by_xpath('.//tr')
                for row in rows:
                    field,values = row.find_elements_by_xpath('.//td')
                    field = field.text[:-1]
                    values = values.text.split('\n')
                    if field in self.categorical_attributes+self.numeric_attributes:
                        plant[field] = values[0]
                    else:
                        for v in values:
                            if f'{field}_{v}' in self.boolean_attributes:
                                plant[f'{field}_{v}'] = True 
                            elif v in self.boolean_attributes:
                                plant[v] = True
                plants = plants.append(plant, ignore_index=True)
        plants[self.boolean_attributes] = plants[self.boolean_attributes].applymap(
                                            lambda x: False if None else True)
        return plants