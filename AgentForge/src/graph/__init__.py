# graph/__init__.py
# 使 graph/ 成为 Python 包，并导出关键符号方便使用。
from graph.state import (
    AgentState,
    InterviewPlan,
    InterviewTopic,
    InterviewQuestion,
    InterviewResult,
    initial_state,
    get_current_topic,
    get_latest_interview_result,
    session_is_complete,
)
# 向后兼容别名
StudyRoadmap = InterviewPlan
Topic = InterviewTopic
QuizResult = InterviewResult
QuizQuestion = InterviewQuestion
