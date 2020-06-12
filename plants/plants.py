import requests
import pandas as pd 


class PlantRecommender:
    """A recommender for designing plant guilds and permaculture gardens.

    Parameters
    ----------
    soil_texture: {'coarse', 'medium', 'fine'}, default='medium'
        Values from the Soil Texture Triangle, {'sand', 'coarse_sand', 
        'fine_sand', 'loamy_coarse_sand', 'loamy_fine_sand', 
        'loamy_very_fine_sand', 'very_fine_sand', 'loamy_sand', 'silt', 
        'sandy_clay_loam', 'very_fine_sandy_loam', 'silty_clay_loam', 
        'silt_loam', 'loam', 'fine_sandy_loam', 'sandy_loam', 
        'coarse_sandy_loam', 'clay_loam', 'sandy_clay', 'silty_clay', 'clay'} 
        may be entered and will be mapped to one of the above.
    
    ph: float, default=6.5
        Soil pH can range from 3.5-8.5. Most plants thrive in soild between 6-7.

    moisture: {'high', 'medium' 'low'}, default='medium'
        A combination of drought tolerance, ability to use soil moisture, and 
        tolerable precipitation.

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
        if soil_texture in {'sand', 'coarse_sand', 'fine_sand', 
                            'loamy_coarse_sand', 'loamy_fine_sand', 
                            'loamy_very_fine_sand', 'very_fine_sand', 
                            'loamy_sand'}:
            self.soil_texture = 'coarse'
        elif soil_texture in {'silt', 'sandy_clay_loam', 'very_fine_sandy_loam', 'silty_clay_loam', 
                            'silt_loam', 'loam', 'fine_sandy_loam', 'sandy_loam', 
                            'coarse_sandy_loam', 'clay_loam'}:
            self.soil_texture = 'medium'
        elif soil_texture in {'sandy_clay', 'silty_clay', 'clay'}:
            self.soil_texture = 'fine'
        else:
            self.soil_texture = soil_texture

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

    

