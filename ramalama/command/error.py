class InferenceEngineError(Exception):
    pass


class InvalidInferenceEngineSpecError(InferenceEngineError):

    def __init__(self, spec_file: str, reason: str, *args):
        super().__init__(*args)

        self.spec_file = spec_file
        self.reason = reason

    def __str__(self):
        return f"Invalid spec file '{self.spec_file}': {self.reason}"


class InferenceEngineSpecMigrationError(InferenceEngineError):

    def __init__(self, spec_file: str, from_version: str, to_version: str, reason: str, *args):
        super().__init__(*args)

        self.spec_file = spec_file
        self.from_version = from_version
        self.to_version = to_version
        self.reason = reason

    def __str__(self):
        return f"Spec migration failed for '{self.spec_file}' from version '{self.from_version}' to '{self.to_version}': {self.reason}"
