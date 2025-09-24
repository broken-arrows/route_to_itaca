#!/usr/bin/env python3
import argparse
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

class CataloniaElectionSimulator:
    """Catalonia Election Simulator using d'Hondt method"""

    def __init__(self, population_file='demographics_data/clean/ok_population_weights_2012.csv', 
                 votes_file='demographics_data/clean/ok_vote_intention_2012.csv'):
        """Initialize with data files"""
        self.population_file = population_file
        self.votes_file = votes_file

        # Catalonia seat allocation by province
        self.seats_per_province = {
            'barcelona': 85,
            'girona': 17, 
            'lleida': 15,
            'tarragona': 18
        }

        self.total_seats = sum(self.seats_per_province.values())

    def load_data(self):
        """Load and validate electoral data"""
        try:
            self.pop_weights = pd.read_csv(self.population_file)
            self.vote_intentions = pd.read_csv(self.votes_file)

            # Normalize province names
            self.pop_weights['province'] = self.pop_weights['province'].str.lower()
            self.vote_intentions['province'] = self.vote_intentions['province'].str.lower()

            print(f"✓ Data loaded successfully!")
            print(f"  Population weights: {self.pop_weights.shape}")
            print(f"  Vote intentions: {self.vote_intentions.shape}")

            return True

        except FileNotFoundError as e:
            print(f"✗ Error loading data files: {e}")
            return False
        except Exception as e:
            print(f"✗ Error processing data: {e}")
            return False

    def calculate_demographic_votes(self):
        """
        Calculate actual votes by combining demographic preferences with population weights.

        This method:
        1. Takes vote intentions by demographic group and province
        2. Weights them by actual demographic distribution  
        3. Excludes abstentions from seat calculations
        4. Returns vote totals by province and party
        """

        # Get all parties (excluding abstain)
        parties = self.vote_intentions[
            self.vote_intentions['party'] != 'abstain'
        ]['party'].unique()

        print(f"  Parties found: {sorted(parties)}")

        # Dictionary to store results
        province_votes = {}

        
        for _, pop_row in self.pop_weights.iterrows():
            province = pop_row['province']
            if province == 'catalunya':  # Skip the overall Catalonia summary row
                continue

            province_votes[province] = {}
            abstention = 0
            total_population = 0

            # Initialize party votes
            for party in parties:
                province_votes[province][party] = 0

            # Calculate weighted votes for each demographic group
            demographic_groups = ['middle', 'retired', 'young', 'unemployed', 'buss', 'rural', 'ind']

            for demo_group in demographic_groups:
                if demo_group not in pop_row or pd.isna(pop_row[demo_group]):
                    continue

                demo_weight = float(pop_row[demo_group])
                total_population += demo_weight

                # Get vote intentions for this province and demographic group
                demo_votes = self.vote_intentions[
                    (self.vote_intentions['province'] == province) & 
                    (self.vote_intentions['demographic_group'] == demo_group)
                ]

                if demo_votes.empty:
                    continue

                # Add weighted votes for each party (excluding abstentions)
                for _, vote_row in demo_votes.iterrows():
                    if vote_row['party'] != 'abstain':
                        party = vote_row['party']
                        # Weight by demographic size and vote percentage
                        weighted_votes = (vote_row['percentage'] / 100) * demo_weight
                        province_votes[province][party] += weighted_votes
                    elif vote_row['party'] == 'abstain':
                        # Abstentions are not counted towards any party, but we want to print them
                        abstention += (vote_row['percentage']/100) * demo_weight

            if total_population > 0:
                abstention_pct = (abstention / total_population) * 100
            else:
                abstention_pct = 0
            print(f"  {province.upper()}: Abstention estimated at {abstention_pct:.1f}%")
        return province_votes

    def apply_threshold(self, province_votes, threshold=3.0):
        """
        Apply electoral threshold per province.

        Args:
            province_votes: Dictionary of votes by province and party
            threshold: Minimum percentage required (default 3.0%)

        Returns:
            Filtered votes dictionary with only parties above threshold
        """
        filtered_votes = {}
        threshold_info = {}

        for province, votes in province_votes.items():
            total_votes = sum(votes.values())
            threshold_votes = (threshold / 100) * total_votes

            filtered_votes[province] = {}
            parties_above_threshold = []
            parties_below_threshold = []

            for party, vote_count in votes.items():
                vote_percentage = (vote_count / total_votes) * 100 if total_votes > 0 else 0

                if vote_count >= threshold_votes:
                    filtered_votes[province][party] = vote_count
                    parties_above_threshold.append((party, vote_percentage))
                else:
                    parties_below_threshold.append((party, vote_percentage))

            threshold_info[province] = {
                'above': parties_above_threshold,
                'below': parties_below_threshold,
                'total_votes': total_votes
            }
        
        return filtered_votes, threshold_info

    def dhondt_allocation(self, votes, seats):
        """
        Allocate seats using d'Hondt method.

        The d'Hondt method divides each party's votes by 1, 2, 3, etc.
        and awards seats to the highest quotients.

        Args:
            votes: Dictionary of party vote totals
            seats: Number of seats to allocate

        Returns:
            Dictionary of seat allocation by party
        """
        if not votes or seats == 0:
            return {party: 0 for party in votes.keys()}

        # Initialize seat allocation
        allocation = {party: 0 for party in votes.keys()}

        # Store allocation history for transparency
        allocation_history = []

        # Allocate seats one by one
        winning_party = None
        for seat_num in range(1, seats + 1):
            max_quotient = 0
            quotients = {}

            # Calculate quotients for all parties
            for party, vote_count in votes.items():
                quotient = vote_count / (allocation[party] + 1)
                quotients[party] = quotient

                if quotient > max_quotient:
                    max_quotient = quotient
                    winning_party = party

            # Award seat to winning party
            if winning_party:
                allocation[winning_party] += 1
                allocation_history.append({
                    'seat': seat_num,
                    'winner': winning_party,
                    'quotient': max_quotient,
                    'all_quotients': quotients.copy()
                })

        if winning_party is not None:
            print(f"\n\n  (last won seat went to {winning_party})")
        else:
            print("\n\n  (no seats were allocated)")
        return allocation, allocation_history

    def simulate_election(self, threshold=3.0, verbose=True):
        """
        Run complete election simulation.

        Args:
            threshold: Electoral threshold percentage (default 3.0%)
            verbose: Whether to print detailed results

        Returns:
            Tuple of (detailed_results, total_results, threshold_info)
        """

        if verbose:
            print("\n" + "=" * 60)
            print("CATALONIA 2012 ELECTION SIMULATION")
            print("=" * 60)
            print(f"Electoral system: d'Hondt method")
            print(f"Threshold: {threshold}% per province")
            print(f"Total seats: {self.total_seats}")
            print()

        # Step 1: Calculate demographic-weighted votes
        province_votes = self.calculate_demographic_votes()

        # Step 2: Apply threshold
        filtered_votes, threshold_info = self.apply_threshold(province_votes, threshold)

        # Step 3: Allocate seats using d'Hondt method
        if verbose:
            print()

        total_results = defaultdict(int)
        detailed_results = {}
        allocation_histories = {}

        if verbose:
            print("RESULTS BY PROVINCE:")
            print("=" * 60)

        for province in ['barcelona', 'girona', 'lleida', 'tarragona']:
            seats = self.seats_per_province[province]
            votes = filtered_votes.get(province, {})

            if votes:
                allocation, history = self.dhondt_allocation(votes, seats)
                detailed_results[province] = allocation
                allocation_histories[province] = history

                if verbose:
                    print(f"\n{province.upper()} ({seats} seats):")
                    print("-" * 40)

                    # Show threshold information
                    threshold_data = threshold_info[province]
                    if threshold_data['below']:
                        print(f"  Parties below {threshold}% threshold:")
                        for party, pct in threshold_data['below']:
                            print(f"    {party.upper()}: {pct:.1f}%")
                        print()

                    # Sort parties by seats won (descending)
                    sorted_parties = sorted(allocation.items(), key=lambda x: x[1], reverse=True)

                    total_votes = sum(votes.values())
                    for party, seats_won in sorted_parties:
                        vote_share = (votes[party] / total_votes) * 100
                        total_results[party] += seats_won
                        print(f"  {party.upper():8}: {seats_won:2} seats ({vote_share:.1f}% votes)")
            else:
                if verbose:
                    print(f"\n{province.upper()}: No parties above threshold")
                detailed_results[province] = {}

        if verbose:
            print("\n" + "=" * 60)
            print("OVERALL PARLIAMENT COMPOSITION:")
            print("=" * 60)

            # Sort by total seats
            sorted_total = sorted(total_results.items(), key=lambda x: x[1], reverse=True)

            total_seats_allocated = sum(total_results.values())

            for party, seats in sorted_total:
                percentage = (seats / self.total_seats) * 100
                print(f"{party.upper():8}: {seats:3} seats ({percentage:.1f}%)")

            print(f"\nTotal seats allocated: {total_seats_allocated}/{self.total_seats}")

            # Show majority information
            majority_threshold = (self.total_seats // 2) + 1
            print(f"Majority threshold: {majority_threshold} seats")

            if sorted_total and sorted_total[0][1] >= majority_threshold:
                print(f"✓ {sorted_total[0][0].upper()} has absolute majority")
            else:
                print("✗ No party has absolute majority - coalition needed")

        return detailed_results, dict(total_results), province_votes

    def export_results(self, detailed_results, total_results, filename='demographics_data/simulation_results/catalonia_2012_results.csv'):
        """Export detailed results to CSV"""

        results_data = []

        # Add province-level results
        for province, allocation in detailed_results.items():
            for party, seats in allocation.items():
                results_data.append({
                    'level': 'province',
                    'area': province,
                    'party': party,
                    'seats': seats,
                    'percentage': (seats / self.seats_per_province[province]) * 100
                })

        # Add overall results
        for party, seats in total_results.items():
            results_data.append({
                'level': 'overall',
                'area': 'catalonia',
                'party': party, 
                'seats': seats,
                'percentage': (seats / self.total_seats) * 100
            })

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filename, index=False)
        print(f"\n✓ Results exported to {filename}")

        return results_df

    def plot_votes_and_seats(self, detailed_results, total_results, province_votes, seats_per_province):
        """
        Visualize votes and seats per province and overall totals.
        """
        party_colors = {
            'ciu': '#002782',
            'cdc': '#002782',
            'unio': '#0052a3',
            'erc': '#ff8000',
            'cup': '#ffed00',
            'jxsi': '#3ab6a5',
            'pdcat': '#0081c2',
            'junts': '#20c0b2',
            'cs': '#f6c300',
            'ppc': '#007fff',
            'pp': '#007fff',
            'psc': '#e73b39',
            'icv': '#67af2f',
            'si': '#000000',
            'csqp': '#c3113b',
            'cecp': '#be3882',
            'ecp': '#6e236e',
            'vox': '#63be21',
            'ac': '#064a81',
            'abstain': '#a0a0a0',
        }

        provinces = list(seats_per_province.keys())
        parties = set()
        for prov in provinces:
            parties.update(province_votes[prov].keys())
        parties = sorted(parties)

        vote_data = {party: [province_votes[prov].get(party, 0) for prov in provinces] for party in parties}
        seat_data = {party: [detailed_results.get(prov, {}).get(party, 0) for prov in provinces] for party in parties}

        _, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Votes per province
        bottom = [0] * len(provinces)
        total_votes_per_prov = [sum(province_votes[prov].values()) for prov in provinces]
        for party in parties:
            bars = axes[0].bar(
                provinces, vote_data[party], bottom=bottom, label=party,
                color=party_colors.get(party, '#888888'), edgecolor='black'
            )
            # Annotate % on each bar segment
            for i, bar in enumerate(bars):
                if vote_data[party][i] > 0 and total_votes_per_prov[i] > 0:
                    pct = 100 * vote_data[party][i] / total_votes_per_prov[i]
                    axes[0].text(
                        bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.1f}%", ha='center', va='center', fontsize=8, color='black', rotation=0
                    )
            bottom = [b + v for b, v in zip(bottom, vote_data[party])]
        axes[0].set_ylabel('Votes')
        axes[0].set_title('Votes per Province (stacked by party)')
        axes[0].legend()

        # Seats per province
        bottom = [0] * len(provinces)
        for party in parties:
            bars = axes[1].bar(
                provinces, seat_data[party], bottom=bottom, label=party,
                color=party_colors.get(party, '#888888'), edgecolor='black'
            )
            for i, bar in enumerate(bars):
                total_seats = seats_per_province[provinces[i]]
                if seat_data[party][i] > 0 and total_seats > 0:
                    pct = seat_data[party][i]
                    axes[1].text(
                        bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.0f}", ha='center', va='center', fontsize=8, color='black', rotation=0
                    )
            bottom = [b + s for b, s in zip(bottom, seat_data[party])]
        axes[1].set_ylabel('Seats')
        axes[1].set_title('Seats per Province (stacked by party)')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        # Total votes and seats
        total_votes = defaultdict(float)
        for prov in provinces:
            for party, votes in province_votes[prov].items():
                total_votes[party] += votes
        total_votes = dict(total_votes)

        sorted_parties_votes = sorted(total_votes.items(), key=lambda x: x[1], reverse=True)
        sorted_parties_seats = sorted(total_results.items(), key=lambda x: x[1], reverse=True)

        _, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Total votes
        total_votes_sum = sum([p[1] for p in sorted_parties_votes])
        bars_votes = axes[0].bar(
            [p[0] for p in sorted_parties_votes], [p[1] for p in sorted_parties_votes],
            color=[party_colors.get(p[0], '#888888') for p in sorted_parties_votes], edgecolor='black'
        )
        for i, bar in enumerate(bars_votes):
            if total_votes_sum > 0:
                pct = 100 * bar.get_height() / total_votes_sum
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f"{pct:.1f}%", ha='center', va='center', fontsize=9, color='black', rotation=0
                )
        axes[0].set_title('Total Votes (Catalonia)')
        axes[0].set_ylabel('Votes')
        axes[0].set_xticklabels([p[0] for p in sorted_parties_votes], rotation=45)

        # Total seats
        total_seats_sum = sum([p[1] for p in sorted_parties_seats])
        bars_seats = axes[1].bar(
            [p[0] for p in sorted_parties_seats], [p[1] for p in sorted_parties_seats],
            color=[party_colors.get(p[0], '#888888') for p in sorted_parties_seats], edgecolor='black'
        )
        for i, bar in enumerate(bars_seats):
            if total_seats_sum > 0:
                pct = bar.get_height()
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f"{pct:.0f}", ha='center', va='center', fontsize=9, color='black', rotation=0
                )
        axes[1].set_title('Total Seats (Catalonia)')
        axes[1].set_ylabel('Seats')
        axes[1].set_xticklabels([p[0] for p in sorted_parties_seats], rotation=45)
        plt.tight_layout()
        plt.show()

def main(population_file: str, votes_file: str, output_file: str):
    """Main execution function"""

    # Initialize simulator
    simulator = CataloniaElectionSimulator(population_file, votes_file)

    # Load data
    if not simulator.load_data():
        return

    print()

    # Run simulation with default 3% threshold

    detailed_results, total_results, province_votes = simulator.simulate_election(
        threshold=3.0, 
        verbose=True
    )

    # Export results
    simulator.export_results(detailed_results, total_results, filename=output_file)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


    # --- Visualization ---
    simulator.plot_votes_and_seats(detailed_results, total_results, province_votes, simulator.seats_per_province)

    return simulator, detailed_results, total_results, province_votes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance demographic percentages in a CSV file.")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="demographics_data/clean/ok_population_weights_2012.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "votes_file",
        nargs="?",
        default="demographics_data/clean/ok_vote_intention_2012.csv",
        help="Path to output CSV file"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="demographics_data/simulation_results/catalonia_2012_results.csv",
        help="Path to output CSV file"
    )
    args = parser.parse_args()
    main(args.input_file, args.votes_file, args.output_file)
