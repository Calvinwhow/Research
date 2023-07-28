import requests
import xml.etree.ElementTree as ET
import numpy as np

# Set the URL for the API query
url = 'http://api.brain-map.org/api/v2/data/query.xml?criteria=model::Donor,rma::criteria,products[abbreviation$eq%27HumanMA%27],rma::include,specimens[parent_id$eqnull](alignment3d),rma::options[only$eq%27donors.id,donors.name,specimens.id,alignment3d.parameters%27]'

# Send a GET request to the URL
response = requests.get(url)

# Parse the XML response
root = ET.fromstring(response.content)

# Loop through each donor in the response
for donor in root.iter('Donor'):
    donor_id = donor.find('id').text
    print('id: \n', donor_id)
    donor_name = donor.find('name').text
    print(donor_name)

    # Loop through each specimen for the donor
    for specimen in donor.iter('Specimen'):
        specimen_id = specimen.find('id').text

        # Check if the specimen has an alignment3d section
        alignment3d = specimen.find('Alignment3d')
        if alignment3d is None:
            continue

        # Extract the transformation matrix from the alignment3d section
        params = alignment3d.find('parameters')
        matrix = np.array(params.text.strip().split(), dtype=np.float32)
        matrix = np.reshape(matrix, (4, 4))

        # Save the matrix to a file
        filename = f'{donor_name}_{specimen_id}_affine.txt'
        np.savetxt(filename, matrix)

        print(f'Saved affine matrix for donor {donor_id}, specimen {specimen_id} to {filename}')
