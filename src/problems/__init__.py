__all__ = ["load_problem_config"]


def load_problem_config(problem_name):
    """
    Dynamically loads a problem class by name.
    Example: 'rocker_arm' -> src.problems.rocker_arm.RockerArmSetup
    """
    try:
        if problem_name == 'rocker_arm':
            from src.problems.rocker_arm import RockerArmSetup
            return RockerArmSetup()
        elif problem_name in ['cantilever', 'top3d_result_YIN_Canteliver_Beam', 'Cantilever_Beam_3D']:
            from src.problems.cantilever import CantileverSetup
            return CantileverSetup()
        elif problem_name == 'generic':
            from src.problems.generic import GenericProblem
            return GenericProblem()
        elif problem_name == 'tagged':
            from src.problems.tagged_problem import TaggedProblem
            return TaggedProblem()
        else:
            raise ValueError(f"Unknown problem: {problem_name}")
    except ImportError as e:
        print(f"[Error] Could not import problem '{problem_name}': {e}")
        return None
