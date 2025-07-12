"""
Test case loading utilities.
"""

import json
from typing import Dict, List
from eval_models import TestCase


class TestCaseLoader:
    """Loads and manages test cases for evaluation."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.test_cases = self.load_test_cases()
    
    def load_test_cases(self) -> Dict[str, List[TestCase]]:
        """Load test cases from the data file."""
        with open(self.config.test_data_path, "r", encoding="utf-8") as f:
            courses_data = json.load(f)

        test_cases = {
            "course_info": [],
            "prerequisites": [],
            "learning_outcomes": [],
            "assessment": []
        }

        for course in courses_data:
            # Course information tests
            test_cases["course_info"].append(TestCase(
                question=f"What is the {course['course_name']} ({course['course_code']}) course about?",
                ground_truth=course['course_content'],
                test_type="course_info",
                course_code=course['course_code'],
                course_name=course['course_name']
            ))

            # Prerequisites tests
            test_cases["prerequisites"].append(TestCase(
                question=f"What are the prerequisites for the course {course['course_name']} ({course['course_code']})?",
                ground_truth=course['entry_requirements'],
                test_type="prerequisites",
                course_code=course['course_code'],
                course_name=course['course_name']
            ))

            # Learning outcomes tests
            if course.get("learning_outcomes"):
                outcomes = course["learning_outcomes"][0]
                formatted_outcomes = (
                    f"Knowledge and Understanding: {outcomes['knowledge_and_understanding']}\n"
                    f"Competence and Skills: {outcomes['competence_and_skills']}\n"
                    f"Judgement and Approach: {outcomes['judgement_and_approach']}"
                )
                test_cases["learning_outcomes"].append(TestCase(
                    question=f"What are the learning outcomes for {course['course_name']} ({course['course_code']})?",
                    ground_truth=formatted_outcomes,
                    test_type="learning_outcomes",
                    course_code=course['course_code'],
                    course_name=course['course_name']
                ))

            # Assessment tests
            test_cases["assessment"].append(TestCase(
                question=f"How is the course {course['course_name']} ({course['course_code']}) assessed?",
                ground_truth=course["assessment"],
                test_type="assessment",
                course_code=course['course_code'],
                course_name=course['course_name']
            ))

        return test_cases
    
    def get_test_cases(self, test_type: str = None, subset_size: int = None) -> List[TestCase]:
        """Get test cases with optional filtering."""
        if test_type:
            cases = self.test_cases.get(test_type, [])
        else:
            # Flatten all test cases
            cases = []
            for type_cases in self.test_cases.values():
                cases.extend(type_cases)
        
        if subset_size:
            cases = cases[:subset_size]
        
        return cases
    
    def get_test_types(self) -> List[str]:
        """Get available test types."""
        return list(self.test_cases.keys())