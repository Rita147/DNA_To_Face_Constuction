import pandas as pd
from .utils import normalize_genotype

class HirisplexPredictor:
    """
    Loads the HirisPlex 41-SNP dictionary CSV and performs
    genotype → phenotype category translation.
    """

    def __init__(self, csv_path="data/hirisplex_full_41snps.csv"):
        self.table = pd.read_csv(csv_path)

    def predict_trait_from_snp(self, rsid, genotype):
        """
        Given rsID and genotype (e.g. 'AG'), returns:
        (trait_name, trait_value) or None if genotype not found.
        """
        genotype = normalize_genotype(genotype)
        df = self.table

        match = df[(df["rsid"] == rsid) & (df["alleles"] == genotype)]
        if match.empty:
            return None

        row = match.iloc[0]
        return row["trait"], row["trait_value"]

    def predict_all(self, snp_dict):
        """
        Takes a dict of rsid → genotype and returns:
        { trait_name: [trait_values...] }
        Multiple SNPs can contribute to the same trait.
        """
        results = {}

        for rsid, genotype in snp_dict.items():
            trait = self.predict_trait_from_snp(rsid, genotype)
            if trait:
                tname, tvalue = trait
                results.setdefault(tname, []).append(tvalue)

        return results
