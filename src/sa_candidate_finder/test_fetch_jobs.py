from sa_candidate_finder.secrets import MANATAL_API_KEY
from sa_candidate_finder.manatal_jobs import fetch_all_jobs

if __name__ == "__main__":
    jobs = fetch_all_jobs(MANATAL_API_KEY)
    print(f"Fetched {len(jobs)} jobs.")
    for job in jobs[:5]:  # Print first 5 jobs for inspection
        print(job)
