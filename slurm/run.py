import sys
import json
import time
import logging
import pickle

import geopandas as gpd
import transitaccess
import gtfs2nx

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    params = json.load(sys.stdin)
    logger.info(f'Running with params: {params}')

    city = params['city']
    gtfs_path = params['gtfs_path']
    target_dir = params['target_dir']
    h3_res = params.get('h3_res')
    agency_ids = params.get('agency_ids')
    route_types = params.get('route_types')
    start_time = params.get('start_time')
    end_time = params.get('end_time')
    boundary_path = params.get('boundary_path')
    frac = params.get('frac')
    crs = params.get('crs')

    timestr = time.strftime('%Y%m%d-%H-%M-%S')
    area = gpd.read_file(boundary_path).to_crs(crs) if boundary_path else None
    boundary = area.dissolve().geometry[0] if area is not None else None

    G = gtfs2nx.transit_graph(
        gtfs_path,
        route_types=route_types,
        time_window=(start_time, end_time),
        agency_ids=agency_ids,
        boundary=boundary,
        frac=frac,
        crs=crs,
    )

    # G = gtfs2nx.utils.closeness_centrality(G)
    # G = gtfs2nx.utils.local_closeness_centrality(G, radius=3000)

    file_path = f'{target_dir}/transit-network-{city}-{timestr}.pkl'

    with open(file_path, 'wb') as handle:
        pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('Transit network graph saved.')

    # with open(file_path, 'rb') as f:
    #     G = pickle.load(f)

    access_score = transitaccess.transit_access_for_neighborhood(G, area, h3_res=h3_res)
    access_score.to_pickle(f'{target_dir}/access-score-{city}-{timestr}.pkl')
    access_score.to_file(f'{target_dir}/access-score-{city}-{timestr}.gpkg', driver='GPKG')

    # access_score_grid = transitaccess.transit_access_for_grid(G, area=None, h3_res=h3_res)
    # access_score_grid.to_pickle(f'{target_dir}/access-score-grid-{city}-{timestr}.pkl')
    # access_score_grid.to_file(f'{target_dir}/access-score-grid-{city}-{timestr}.gpkg', driver='GPKG')
    # logger.info('Gridded transit network accessibility scores saved.')
