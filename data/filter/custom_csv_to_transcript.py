import argparse
import csv
import re

    

def clean_pgn(pgn_string):
    # Remove headers

    pgn_string = re.sub(r'\[.*?\]\s*', '', pgn_string)
    # Remove clock annotations
    pgn_string = re.sub(r'\{\}', '', pgn_string)
    
    # Replace numbered moves with correct formatting
    pgn_string = re.sub(r'\b(\d+)\.\s+', r'\1.', pgn_string)
    
    # Remove duplicate numbering of black's move
    pgn_string = re.sub(r'(\d+\.\S+)\s+\d+\.\.\.', r'\1', pgn_string)
    pgn_string = re.sub(r'\s+', ' ', pgn_string).strip()
    pgn_string = re.sub(r'\s*(1-0|0-1|1/2-1/2)\s*$', '', pgn_string)
 
    return pgn_string

def filter_games(input_file, output_file, min_elo, max_elo):
    selected_games = 0
    filtered_games = 0
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = ['transcript']  # Only keeping the transcript column in the output
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            try:
                white_elo = int(row['WhiteElo'])
                black_elo = int(row['BlackElo'])
                
                if min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo:
                    
                    transcript = row.get('pgn', '')
                    clean_transcript = clean_pgn(transcript)

                    writer.writerow({'transcript': clean_transcript[:1023]})
                    selected_games += 1
                else:
                    filtered_games += 1
            except ValueError:
                # Skip rows with invalid Elo ratings
                filtered_games += 1
                continue
    
    print(f"Total games processed: {selected_games + filtered_games}")
    print(f"Games selected: {selected_games}")
    print(f"Games filtered out: {filtered_games}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter chess games by Elo ratings.")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file")
    parser.add_argument("--minelo", type=int, help="Minimum allowed Elo rating")
    parser.add_argument("--maxelo", type=int, help="Maximum allowed Elo rating")
    
    args = parser.parse_args()
    
    filter_games(args.input_file, args.output_file, args.minelo, args.maxelo)
