import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import argparse


class Issue(object):
    def __init__(self, name, stances):
        """
        name: str - name of the issue
        stances: list[str] - list of all possible stances that could be taken on this issue
        """
        self.name = name
        self.stances = stances
        self.vote_totals = {s: 0 for s in stances}

    def count_issue_vote(self, vote):
        self.vote_totals[vote] += 1

    def print_vote_totals(self):
        print(self.vote_totals)


class IssueVotes(object):
    def __init__(self, issue):
        self.issue = issue
        self.stance_votes = {s: 0 for s in issue.stances}

    def get_majority_stance(self):
        biggest = 0
        maj_stance = None
        for k, v in self.stance_votes.items():
            if v > biggest:
                maj_stance = k
                biggest = v
        return maj_stance

    def vote_for_stance(self, stance):
        self.stance_votes[stance] += 1


class View(object):
    def __init__(self, issue, stance, weight=1):
        """

        :param issue: Issue
        :param stance:
        """
        self.issue = issue
        self.stance = stance
        self.weight = weight


class PopulationView(object):
    def __init__(self, issue, proportions, weight, weight_variance):
        """

        :param issue: an Issue object
        :param proportions: dict{str: float} - key is a str stance on an issue and value is proportion in the population
        holding that stance on this issue
        :param weight: float representing mean importance of the issue to people in this population
        :param weight_variance: float variance in importance of the issue to people in this population
        """
        self.issue = issue
        self.proportions = proportions
        self.weight = weight
        self.weight_variance = weight_variance


class Person(object):
    def __init__(self, population_views):
        """

        :param population_views: list[PopulationView]
        """
        self.views = []
        self._decide_views(population_views)
        # it's important for later comparison to make the views be in order by issue name
        self.views = sorted(self.views, key=lambda x: x.issue.name)

    def _decide_views(self, population_views):
        """

        :param population_views: list[PopulationView]
        :return: None
        """
        for i in population_views:
            issue = i.issue
            possible_stances = issue.stances
            probabilities = [i.proportions[j] for j in possible_stances]
            choice = np.random.choice(possible_stances, p=probabilities)
            mean_importance = i.weight
            std_importance = i.weight_variance
            # max makes sure we don't have below 0 importance values
            importance = max(0, np.random.normal(mean_importance, std_importance, 1)[0])
            self.views.append(View(issue, choice, importance))

    def vote(self, candidates):
        agreements = []
        # figuring out on how many issues this person agreements with each candidate
        for c in candidates:
            c_agreements = {'candidate': c, 'agreements': 0}
            for i, j in zip(c.views, self.views):
                if i.issue.name != j.issue.name:
                    raise Exception
                if i.stance == j.stance:
                    c_agreements['agreements'] += 1 * j.weight
            agreements.append(c_agreements)

        # sort candidates by how many (weighted) issues they agree with person and take top candidate
        agreements = sorted(agreements, key=lambda x: x['agreements'], reverse=True)
        preferred_candidate = agreements[0]['candidate']

        # vote for the top candidate
        preferred_candidate.receive_vote()


class Candidate(object):
    def __init__(self, name, views):
        self.views = views
        self.votes = 0
        self.name = name
        # make sure views are sorted by issue name for later comparison
        self.views = sorted(self.views, key=lambda x: x.issue.name)

    def receive_vote(self):
        self.votes += 1


class Population(object):
    def __init__(self, name, size, population_views):
        """
        :param name: str name of this population e.g. "left"
        :param size: int number of people in this population
        population_views: list[PopulationView] views of this population
        """
        self.size = size
        self.name = name
        self.population_views = population_views
        # create the people who are members of this population (Person objects)
        self.people = self._generate_people(self.population_views)

    def _generate_people(self, population_views):
        people = []
        for _ in tqdm(range(self.size)):
            people.append(Person(population_views))
        return people


def generate_populations(pop_settings, issues):
    """

    :param pop_settings: list of dicts describing populations for an election
    [{'name': str population name,
      'size': int population size,
      'issue_views': [
        {'name': str name of issue
        'weight': float mean importance of the issue
        'weight_variation': float std dev of importance of the issue
        'stances': {
          'stance_1': float proportion in this population having stance_1,
          'stance_2': float proportion in this population having stance_2,
          ...
          }
        ]
      },
      ...
    ]
    :param issues: list[Issue] the issues in this election, sorted by name
    :return: list[Population]
    """
    populations = []
    for pop in pop_settings:
        pop_views = []
        # make sure issue_views are sorted by name for next line where we zip them with issue list (also sorted by name)
        issue_views = sorted(pop['issue_views'], key=lambda x: x['name'])
        for v, issue in zip(issue_views, issues):
            pop_views.append(PopulationView(issue=issue,
                                            proportions=v['stances'],
                                            weight=v['weight'],
                                            weight_variance=v['weight_variance']))

        populations.append(Population(pop['name'], pop['size'], pop_views))

    return populations


def generate_issues(issue_settings):
    """

    :param issue_settings: list of dicts describing issues for an election
    [{'name': str issue name,
      'stances': list[str] names of all possible stances for this issue
     },
     ...
    ]
    :return: list[Issue]
    """
    issues = []
    for i in issue_settings:
        issues.append(Issue(i['name'], i['stances']))

    # make sure issue list is sorted by name--important for later comparison
    issues = sorted(issues, key=lambda x: x.name)
    return issues


def generate_candidates(candidate_settings, issues):
    """

    :param candidate_settings: list of dicts describing candidates for an election
    [{'name': str candidate name,
      'views': [
        {'issue': str name of issue
         'stance': str name of stance this candidate holds on this issue
        },
        ...
      ]
    ]
    :return:
    """
    candidates = []
    for c in candidate_settings:
        cviews = sorted(c['views'], key=lambda x: x['issue'])
        views = []
        for cview, issue in zip(cviews, issues):
            views.append(View(issue, cview['stance']))
        candidates.append(Candidate(c['name'], views))
    return candidates


def analyze_results(people, candidate):
    # determine distribution of how many issues voters agreed with the winner on
    people_agreements_with_cand = []
    for person in people:
        agreements = 0
        for pv, cv in zip(person.views, candidate.views):
            if pv.stance == cv.stance:
                agreements += 1
        people_agreements_with_cand.append(agreements)
    df = pd.Series(people_agreements_with_cand)
    print('distribution of agreement with candidate', candidate.name)
    print(df.describe())


def generate_majority_candidate(people, issues):
    """

    :param people: list[Person]
    :param issues: list[Issue]
    :return:
    """
    issue_votes = [IssueVotes(i) for i in issues]

    for p in people:
        for v, iv in zip(p.views, issue_votes):
            iv.vote_for_stance(v.stance)

    maj_views = []
    for i, iv in zip(issues, issue_votes):
        maj_views.append(View(i, iv.get_majority_stance()))

    return Candidate('majority', maj_views)


def run_election(settings_filepath):
    """

    :param settings_filepath: str filepath to a json settings file
    :return: None
    """
    settings = json.load(open(settings_filepath, 'r'))

    issues = generate_issues(settings['issues'])
    candidates = generate_candidates(settings['candidates'], issues)
    populations = generate_populations(settings['populations'], issues)

    all_people = []
    for pop in populations:
        for person in pop.people:
            person.vote(candidates)
            all_people.append(person)



    candidates = sorted(candidates, key=lambda x: x.votes, reverse=True)

    for c in candidates:
        print(c.name, c.votes)

    winner = candidates[0]
    loser = candidates[-1]

    print('winner is ', winner.name)
    print('loser is ', loser.name)

    for person in all_people:
        for view in person.views:
            view.issue.count_issue_vote(view.stance)

    for i in issues:
        i.print_vote_totals()

    analyze_results(all_people, winner)
    analyze_results(all_people, loser)

    majority_candidate = generate_majority_candidate(all_people, issues)
    analyze_results(all_people, majority_candidate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a simulated election.')
    parser.add_argument('in_file', type=str, help='json file describing the election')
    args = parser.parse_args()
    run_election(args.in_file)
    # run_election('../data/wedge_high_agreement_settings.json')
